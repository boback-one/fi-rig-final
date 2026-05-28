"""
fi_host.transport.serial_driver
Async-capable serial driver for the ESP32-S3 rig.
Supports both sync (CLI) and async (FastAPI/WebSocket) use.
"""
from __future__ import annotations
import asyncio
import json
import threading
import time
from typing import AsyncIterator, Iterator, Optional

import serial
import serial.tools.list_ports

from fi_host.core import GlitchParams, SweepParams, GlitchRecord, SessionStats

DEFAULT_BAUD    = 921600
RESP_TIMEOUT    = 5.0
CONNECT_TIMEOUT = 3.0


class RigConnectionError(Exception): ...
class RigTimeoutError(Exception): ...


# ─────────────────────────── SYNC DRIVER ────────────────────────────────────

class RigSerial:
    """
    Synchronous serial driver. Use in CLI / threading contexts.
    Thread-safe: all public methods acquire _lock.
    """

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self.port  = port
        self.baud  = baud
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self.stats = SessionStats()

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                self.port, self.baud,
                timeout       = RESP_TIMEOUT,
                write_timeout = 2.0,
            )
            time.sleep(0.15)
            self._ser.reset_input_buffer()
            resp = self._cmd("STATUS")
            return resp is not None and resp.get("status") == "ready"
        except serial.SerialException as e:
            raise RigConnectionError(f"Could not open {self.port}: {e}") from e

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        self.connect(); return self

    def __exit__(self, *_):
        self.disconnect()

    # ── low level ───────────────────────────────────────────────────────────

    def _write(self, cmd: str):
        self._ser.write((cmd + "\n").encode())
        self._ser.flush()

    def _readline(self) -> Optional[str]:
        try:
            return self._ser.readline().decode("utf-8", errors="replace").strip() or None
        except serial.SerialTimeoutException:
            return None

    def _cmd(self, cmd: str) -> Optional[dict]:
        with self._lock:
            self._write(cmd)
            line = self._readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {"raw": line}

    # ── public API ──────────────────────────────────────────────────────────

    def reset_target(self) -> bool:
        resp = self._cmd("RESET")
        return resp is not None and resp.get("status") == "reset_ok"

    def firmware_status(self) -> Optional[dict]:
        return self._cmd("STATUS")

    def glitch_once(self, params: GlitchParams) -> Optional[GlitchRecord]:
        with self._lock:
            self._write(params.to_firmware_cmd())
            line = self._readline()
        if not line:
            return None
        rec = GlitchRecord.from_json_line(line)
        if rec:
            self.stats.record(rec)
        return rec

    def sweep_iter(self, params: SweepParams) -> Iterator[GlitchRecord]:
        total = params.total_combinations
        with self._lock:
            self._write(params.to_firmware_cmd())
            received = 0
            while received < total:
                line = self._readline()
                if line is None:
                    break
                if '"sweep_done"' in line:
                    break
                rec = GlitchRecord.from_json_line(line)
                if rec:
                    rec.attempt = received
                    received   += 1
                    self.stats.record(rec)
                    yield rec

    # ── utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def list_ports() -> list[dict]:
        return [
            {"device": p.device, "description": p.description, "hwid": p.hwid}
            for p in serial.tools.list_ports.comports()
        ]


# ─────────────────────────── ASYNC DRIVER ───────────────────────────────────

class AsyncRigSerial:
    """
    Async wrapper — offloads blocking serial I/O to a thread pool.
    Safe for use in FastAPI / asyncio contexts.
    """

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self._sync = RigSerial(port, baud)

    async def connect(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.connect)

    async def disconnect(self):
        self._sync.disconnect()

    async def firmware_status(self) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.firmware_status)

    async def reset_target(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.reset_target)

    async def glitch_once(self, params: GlitchParams) -> Optional[GlitchRecord]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.glitch_once, params)

    async def sweep_stream(self, params: SweepParams) -> AsyncIterator[GlitchRecord]:
        """Yields GlitchRecord objects as they arrive from the rig."""
        loop   = asyncio.get_running_loop()
        queue:  asyncio.Queue[Optional[GlitchRecord]] = asyncio.Queue()

        def _blocking():
            try:
                for rec in self._sync.sweep_iter(params):
                    asyncio.run_coroutine_threadsafe(queue.put(rec), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=_blocking, daemon=True)
        thread.start()

        while True:
            rec = await queue.get()
            if rec is None:
                break
            yield rec

    @property
    def stats(self) -> SessionStats:
        return self._sync.stats

    @staticmethod
    def list_ports() -> list[dict]:
        return RigSerial.list_ports()
