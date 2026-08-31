"""
adaptive_signal.py
------------------
Adaptive Traffic Signal Controller.
Calculates dynamic green light times based on real-time lane densities, queue lengths,
and cumulative waiting times provided by the Digital Twin.
"""

from typing import Dict, Tuple, Optional
import traci
from ..digital_twin.digital_twin_system import DigitalTwin

class AdaptiveSignalController:
    """
    Dynamically adjusts traffic signal phases and green durations based on live traffic demand.
    """

    def __init__(
        self,
        traffic_light_id: str,
        phase_map: Dict[str, int],
        min_green: int = 15,
        max_green: int = 60,
        base_green: int = 25
    ):
        self.traffic_light_id = traffic_light_id
        self.phase_map = phase_map
        self.min_green = min_green
        self.max_green = max_green
        self.base_green = base_green

    def compute_green_time(self, twin: DigitalTwin, lane_id: str, predicted_count: Optional[float] = None) -> int:
        """
        Calculates optimal green duration for the given lane using a weighted demand function:
        - Vehicle Count: +4s per 2 vehicles
        - Queue Length: +3s per halted vehicle
        - Average Waiting Time: +2s per 10s wait
        - AI Predicted Surge: +3s per 2 predicted incoming vehicles (proactive adjustment)
        Bounded strictly between [min_green, max_green].
        """
        lane = twin.lanes.get(lane_id)
        if not lane:
            return self.base_green

        # Demand scoring
        veh_factor = (lane.vehicle_count // 2) * 4
        queue_factor = lane.queue_length * 3
        wait_factor = int(lane.waiting_time // 10) * 2

        # AI Proactive Surge Factor
        ai_factor = 0
        if predicted_count is not None:
            ai_factor = int(predicted_count // 2) * 3

        calculated_time = self.base_green + veh_factor + queue_factor + wait_factor + ai_factor
        bounded_time = max(self.min_green, min(self.max_green, calculated_time))

        return bounded_time

    def execute_adaptive_cycle(
        self,
        twin: DigitalTwin,
        lane_predictions: Optional[Dict[str, Tuple[float, str]]] = None
    ) -> Tuple[str, int, int]:
        """
        Runs one adaptive control cycle:
        1. Finds highest-demand approach (considering live state + AI predictions).
        2. Computes proactive adaptive green time.
        3. Pushes phase and duration to SUMO via TraCI.

        Returns:
            (selected_lane_id: str, target_phase: int, allocated_green_time: int)
        """
        # Determine highest priority lane
        target_lane_id = twin.get_densest_approach()
        target_phase = self.phase_map.get(target_lane_id, 0)

        pred_count = None
        if lane_predictions and target_lane_id in lane_predictions:
            pred_count = lane_predictions[target_lane_id][0]

        green_time = self.compute_green_time(twin, target_lane_id, predicted_count=pred_count)

        # Update Digital Twin Signal State
        twin.signal.current_phase = target_phase
        twin.signal.current_duration = green_time

        # Push to SUMO if simulation is active
        try:
            if traci.isLoaded():
                traci.trafficlight.setPhase(self.traffic_light_id, target_phase)
                traci.trafficlight.setPhaseDuration(self.traffic_light_id, green_time)
                print(
                    f"[ADAPTIVE SIGNAL] Lane {target_lane_id} (Vehicles={twin.lanes[target_lane_id].vehicle_count}, "
                    f"Queue={twin.lanes[target_lane_id].queue_length}) -> Phase {target_phase}, Green Time: {green_time}s"
                )
        except Exception as err:
            print(f"[ERROR] Failed to push adaptive signal: {err}")

        return target_lane_id, target_phase, green_time
