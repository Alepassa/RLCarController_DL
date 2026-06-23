"""Assetto Corsa shared memory reader.

Three memory-mapped files exposed by AC:
  - Local\\acpmf_physics    (~333 Hz physics)
  - Local\\acpmf_graphics   (~60 Hz HUD/state)
  - Local\\acpmf_static     (session-static info)

Only the fields we need are declared; trailing bytes are padded.
Layouts based on the official ACPMF reference.
"""
import ctypes
import mmap
from ctypes import c_int, c_float, c_wchar


class SPageFilePhysics(ctypes.Structure):
    _fields_ = [
        ("packetId", c_int),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),
        ("gear", c_int),
        ("rpms", c_int),
        ("steerAngle", c_float),
        ("speedKmh", c_float),
        ("velocity", c_float * 3),
        ("accG", c_float * 3),
        ("wheelSlip", c_float * 4),
        ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4),
        ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4),
        ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4),
        ("drs", c_float),
        ("tc", c_float),
        ("heading", c_float),
        ("pitch", c_float),
        ("roll", c_float),
        ("cgHeight", c_float),
        ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int),
        ("pitLimiterOn", c_int),
        ("abs", c_float),
        ("_pad", c_float * 64),
    ]


class SPageFileGraphics(ctypes.Structure):
    _fields_ = [
        ("packetId", c_int),
        ("status", c_int),
        ("session", c_int),
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int),
        ("position", c_int),
        ("iCurrentTime", c_int),
        ("iLastTime", c_int),
        ("iBestTime", c_int),
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int),
        ("currentSectorIndex", c_int),
        ("lastSectorTime", c_int),
        ("numberOfLaps", c_int),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),
        ("carCoordinates", c_float * 3),
        ("_pad", c_float * 128),
    ]


class SPageFileStatic(ctypes.Structure):
    _fields_ = [
        ("smVersion", c_wchar * 15),
        ("acVersion", c_wchar * 15),
        ("numberOfSessions", c_int),
        ("numCars", c_int),
        ("carModel", c_wchar * 33),
        ("trackName", c_wchar * 33),
        ("trackConfig", c_wchar * 33),
        ("maxRpm", c_int),
        ("maxFuel", c_float),
        ("suspensionMaxTravel", c_float * 4),
        ("tyreRadius", c_float * 4),
        ("_pad", c_float * 64),
    ]


class SharedMemoryReader:
    def __init__(self):
        self._mm_p = mmap.mmap(-1, ctypes.sizeof(SPageFilePhysics), "Local\\acpmf_physics")
        self._mm_g = mmap.mmap(-1, ctypes.sizeof(SPageFileGraphics), "Local\\acpmf_graphics")
        self._mm_s = mmap.mmap(-1, ctypes.sizeof(SPageFileStatic), "Local\\acpmf_static")

    def read_physics(self) -> SPageFilePhysics:
        self._mm_p.seek(0)
        return SPageFilePhysics.from_buffer_copy(self._mm_p.read(ctypes.sizeof(SPageFilePhysics)))

    def read_graphics(self) -> SPageFileGraphics:
        self._mm_g.seek(0)
        return SPageFileGraphics.from_buffer_copy(self._mm_g.read(ctypes.sizeof(SPageFileGraphics)))

    def read_static(self) -> SPageFileStatic:
        self._mm_s.seek(0)
        return SPageFileStatic.from_buffer_copy(self._mm_s.read(ctypes.sizeof(SPageFileStatic)))

    def wait_new_physics(self, last_packet_id: int, timeout_s: float = 0.5) -> SPageFilePhysics:
        import time
        deadline = time.perf_counter() + timeout_s
        while True:
            p = self.read_physics()
            if p.packetId != last_packet_id:
                return p
            if time.perf_counter() > deadline:
                return p

    def close(self):
        self._mm_p.close()
        self._mm_g.close()
        self._mm_s.close()
