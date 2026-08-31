"""
signal_twin.py
--------------
Virtual representation of a Traffic Light Junction (Signal Twin) in the Digital Twin.
Mirrors active phase, elapsed green time, target green duration, and signal states.
"""

from typing import Dict, Optional
import traci

class SignalTwin:
    """Virtual representation of a traffic light controller (e.g. TL1)."""

    def __init__(self, tl_id: str, default_phase_names: Optional[Dict[int, str]] = None):
        self.tl_id = tl_id
        self.current_phase = 0
        self.phase_state = ""               # e.g., "GGggrrrr"
        self.current_duration = 30          # Active planned duration in seconds
        self.time_in_phase = 0              # Seconds spent in current phase
        self.is_overridden = False          # True if emergency preemption is currently active
        self.override_reason = ""

        # Human-readable labels for phases (e.g., 0: "North-South Green", 2: "East-West Green")
        self.phase_names = default_phase_names or {
            0: "North-South Green",
            1: "North-South Yellow",
            2: "East-West Green",
            3: "East-West Yellow",
        }

    def update_from_traci(self):
        """Pull live traffic signal status from TraCI."""
        try:
            self.current_phase = traci.trafficlight.getPhase(self.tl_id)
            self.phase_state = traci.trafficlight.getRedYellowGreenState(self.tl_id)
            # spent duration is updated incrementally or queried via next switch time
        except Exception as err:
            print(f"[ERROR] Failed to update SignalTwin for {self.tl_id}: {err}")

    def update_mock(self, phase: int, state: str, duration: int, is_overridden: bool = False, reason: str = ""):
        """Manually update state for tests or offline execution."""
        self.current_phase = phase
        self.phase_state = state
        self.current_duration = duration
        self.is_overridden = is_overridden
        self.override_reason = reason

    def to_dict(self) -> dict:
        """Serialize signal state for WebSocket / API dashboards."""
        return {
            "traffic_light_id": self.tl_id,
            "current_phase": self.current_phase,
            "phase_name": self.phase_names.get(self.current_phase, f"Phase {self.current_phase}"),
            "phase_state": self.phase_state,
            "duration": self.current_duration,
            "is_overridden": self.is_overridden,
            "override_reason": self.override_reason,
        }

    def __repr__(self) -> str:
        name = self.phase_names.get(self.current_phase, f"Phase {self.current_phase}")
        override_str = f" [OVERRIDE: {self.override_reason}]" if self.is_overridden else ""
        return f"SignalTwin({self.tl_id}: phase={self.current_phase} ({name}), state={self.phase_state}{override_str})"
