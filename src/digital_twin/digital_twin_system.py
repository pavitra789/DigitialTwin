"""
digital_twin_system.py
----------------------
Core Digital Twin aggregator.
Coordinates LaneTwins, VehicleTwins, and SignalTwins, maintaining the synchronized
in-memory model of the physical/simulated intersection.
"""

from typing import List, Dict, Optional
import time
import traci

from .lane_twin import LaneTwin
from .vehicle_twin import VehicleTwin
from .signal_twin import SignalTwin

class DigitalTwin:
    """
    Central Digital Twin system.
    Maintains a mirror of all monitored lanes, active vehicles, and traffic lights.
    """

    def __init__(self, lane_ids: List[str], traffic_light_id: str = "TL1", emergency_type: str = "emergency"):
        self.traffic_light_id = traffic_light_id
        self.emergency_type = emergency_type
        self.step_count = 0
        self.timestamp = time.time()

        # Component Twins
        self.lanes: Dict[str, LaneTwin] = {
            lid: LaneTwin(lid, emergency_type=emergency_type) for lid in lane_ids
        }
        self.vehicles: Dict[str, VehicleTwin] = {}
        self.signal = SignalTwin(traffic_light_id)

    def sync_from_traci(self, current_step: int = 0):
        """
        Synchronizes the entire Digital Twin state from SUMO via TraCI.
        Called once per simulation step.
        """
        self.step_count = current_step
        self.timestamp = time.time()

        # 1. Sync Lanes
        for lane in self.lanes.values():
            lane.update_from_traci()

        # 2. Sync Vehicles
        active_veh_ids = set()
        for lane in self.lanes.values():
            for vid in lane.active_vehicle_ids:
                active_veh_ids.add(vid)
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    speed = float(traci.vehicle.getSpeed(vid))
                    pos = traci.vehicle.getPosition(vid)
                    lane_id = traci.vehicle.getLaneID(vid)
                    distance = float(traci.vehicle.getDistance(vid))
                    wait_time = float(traci.vehicle.getWaitingTime(vid))

                    if vid not in self.vehicles:
                        self.vehicles[vid] = VehicleTwin(vid, vehicle_type=v_type)

                    self.vehicles[vid].update(speed, pos, lane_id, distance, wait_time)
                except Exception:
                    pass

        # Prune vehicles that left the monitored intersection
        self.vehicles = {vid: v for vid, v in self.vehicles.items() if vid in active_veh_ids}

        # 3. Sync Signal
        self.signal.update_from_traci()

    def get_state_snapshot(self) -> dict:
        """
        Returns a complete serializable JSON-ready snapshot of the current Digital Twin.
        Used by the FastAPI WebSocket push and AI prediction feature store.
        """
        total_vehicles = sum(l.vehicle_count for l in self.lanes.values())
        total_queue = sum(l.queue_length for l in self.lanes.values())
        avg_speed = (
            sum(l.avg_speed for l in self.lanes.values()) / len(self.lanes)
            if self.lanes else 0.0
        )

        return {
            "step": self.step_count,
            "timestamp": self.timestamp,
            "summary": {
                "total_vehicles": total_vehicles,
                "total_queue": total_queue,
                "avg_speed": round(avg_speed, 2),
                "emergency_active": any(l.has_emergency_vehicle for l in self.lanes.values()),
            },
            "lanes": {lid: lt.to_dict() for lid, lt in self.lanes.items()},
            "signal": self.signal.to_dict(),
            "active_vehicle_count": len(self.vehicles),
        }

    def get_emergency_lane(self) -> Optional[str]:
        """Returns the lane ID of the first lane detected with an emergency vehicle, if any."""
        for lane_id, lane in self.lanes.items():
            if lane.has_emergency_vehicle:
                return lane_id
        return None

    def get_densest_approach(self) -> str:
        """Returns the lane ID with the highest vehicle density/queue score."""
        return max(
            self.lanes.values(),
            key=lambda lt: (lt.vehicle_count * 2 + lt.queue_length * 3 + lt.waiting_time * 0.1)
        ).lane_id
