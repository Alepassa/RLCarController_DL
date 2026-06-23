"""Thin wrapper over pyvjoy for steer/throttle/brake.

Action convention:
  steer ∈ [-1, +1]  → vJoy X axis
  throttle ∈ [-1, +1] -> positive = gas (Y axis), negative = brake (Z axis).
                        Never both at once (mutually exclusive by sign).
"""
import pyvjoy

VJOY_MAX = 0x8000  # pyvjoy axis range: 0 .. 0x8000


class VJoyController:
    def __init__(self, device_id: int = 1):
        self.dev = pyvjoy.VJoyDevice(device_id)
        self.set(0.0, 0.0)

    @staticmethod
    def _to_axis(x: float) -> int:
        x = max(-1.0, min(1.0, x))
        return int((x + 1.0) * 0.5 * VJOY_MAX)

    @staticmethod
    def _to_axis_unipolar(x: float) -> int:
        x = max(0.0, min(1.0, x))
        return int(x * VJOY_MAX)

    def set(self, steer: float, throttle: float) -> None:
        self.dev.set_axis(pyvjoy.HID_USAGE_X, self._to_axis(steer))
        gas = max(0.0, throttle)
        brake = max(0.0, -throttle)
        self.dev.set_axis(pyvjoy.HID_USAGE_Y, self._to_axis_unipolar(gas))
        self.dev.set_axis(pyvjoy.HID_USAGE_Z, self._to_axis_unipolar(brake))

    def neutral(self) -> None:
        self.set(0.0, 0.0)
