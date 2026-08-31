"""
emergency_priority.py
----------------------
Emergency Vehicle Priority Controller.
Scans the Digital Twin state and provides preemption override:
When an emergency vehicle (ambulance, fire truck) is detected approaching
the intersection, normal adaptive cycle is paused and immediate Green Wave
is granted to the corresponding direction.
"""

from typing import Dict, Optional, Tuple
import traci
from ..digital_twin.digital_twin_system import DigitalTwin

class EmergencyPriorityController:
    """
    Handles preemption and signal override for emergency vehicles.
    """

    def __init__(
        self,
        traffic_light_id: str,
        phase_map: Dict[str, int],
        emergency_green_time: int = 60
    ):
        self.traffic_light_id = traffic_light_id
        self.phase_map = phase_map
        self.emergency_green_time = emergency_green_time

    def check_and_apply_priority(self, twin: DigitalTwin) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Evaluates emergency vehicle presence in the Digital Twin.
        If detected, immediately overrides the traffic light phase to green for that approach.

        Returns:
            (is_active: bool, triggered_lane: Optional[str], applied_phase: Optional[int])
        """
        emergency_lane = twin.get_emergency_lane()

        if emergency_lane is None:
            # Clear override state in signal twin
            if twin.signal.is_overridden:
                twin.signal.is_overridden = False
                twin.signal.override_reason = ""
            return False, None, None

        target_phase = self.phase_map.get(emergency_lane)
        if target_phase is None:
            print(f"[WARN] Emergency vehicle on {emergency_lane} has no mapped phase!")
            return False, emergency_lane, None

        # Update Digital Twin Signal state
        twin.signal.is_overridden = True
        twin.signal.override_reason = f"Emergency Vehicle detected in {emergency_lane}"
        twin.signal.current_duration = self.emergency_green_time

        # Push override to SUMO via TraCI if simulation is active
        try:
            if traci.isLoaded():
                traci.trafficlight.setPhase(self.traffic_light_id, target_phase)
                traci.trafficlight.setPhaseDuration(self.traffic_light_id, self.emergency_green_time)
                print(f"[EMERGENCY PRIORITY] *** OVERRIDE ACTIVE *** -> Lane: {emergency_lane}, Phase: {target_phase}, Duration: {self.emergency_green_time}s")
        except Exception as err:
            print(f"[ERROR] Could not apply emergency signal override: {err}")

        return True, emergency_lane, target_phase
