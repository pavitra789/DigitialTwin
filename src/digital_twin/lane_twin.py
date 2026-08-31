"""
lane_twin.py
------------
Virtual representation of a single traffic lane in the Digital Twin.
Aggregates vehicle count, average speed, queue length, and emergency detection.
"""

from typing import List, Optional
import traci

class LaneTwin:
    """Virtual representation of one lane's real-time state."""

    def __init__(self, lane_id: str, emergency_type: str = "emergency"):
        self.lane_id = lane_id
        self.emergency_type = emergency_type

        # Real-time state metrics
        self.vehicle_count = 0
        self.avg_speed = 0.0          # Average speed of vehicles in this lane (m/s)
        self.queue_length = 0         # Number of stopped/halting vehicles (speed < 0.1 m/s)
        self.waiting_time = 0.0       # Cumulative waiting time in seconds
        self.occupancy = 0.0          # Lane occupancy percentage (0 - 100%)
        self.has_emergency_vehicle = False
        self.emergency_vehicle_ids: List[str] = []
        self.active_vehicle_ids: List[str] = []

    def update_from_traci(self):
        """Pull live lane telemetry from TraCI in active SUMO simulation."""
        try:
            self.active_vehicle_ids = list(traci.lane.getLastStepVehicleIDs(self.lane_id))
            self.vehicle_count = len(self.active_vehicle_ids)
            self.avg_speed = float(traci.lane.getLastStepMeanSpeed(self.lane_id))
            self.queue_length = int(traci.lane.getLastStepHaltingNumber(self.lane_id))
            self.waiting_time = float(traci.lane.getWaitingTime(self.lane_id))
            self.occupancy = float(traci.lane.getLastStepOccupancy(self.lane_id)) * 100.0

            # Scan for emergency vehicles (e.g. ambulances)
            self.emergency_vehicle_ids = [
                vid for vid in self.active_vehicle_ids
                if traci.vehicle.getTypeID(vid) == self.emergency_type
            ]
            self.has_emergency_vehicle = len(self.emergency_vehicle_ids) > 0

        except Exception as err:
            print(f"[ERROR] Failed to update LaneTwin for {self.lane_id}: {err}")

    def update_mock(self, vehicle_count: int, avg_speed: float, queue_length: int, waiting_time: float, has_emergency: bool = False):
        """Manually update state for offline simulation, unit tests, or fallback mode."""
        self.vehicle_count = vehicle_count
        self.avg_speed = avg_speed
        self.queue_length = queue_length
        self.waiting_time = waiting_time
        self.has_emergency_vehicle = has_emergency
        self.occupancy = min(100.0, vehicle_count * 15.0)

    def to_dict(self) -> dict:
        """Serialize lane metrics to a dictionary."""
        return {
            "lane_id": self.lane_id,
            "vehicle_count": self.vehicle_count,
            "avg_speed": round(self.avg_speed, 2),
            "queue_length": self.queue_length,
            "waiting_time": round(self.waiting_time, 2),
            "occupancy": round(self.occupancy, 2),
            "has_emergency_vehicle": self.has_emergency_vehicle,
            "emergency_vehicles": self.emergency_vehicle_ids,
        }

    def __repr__(self) -> str:
        return (
            f"LaneTwin({self.lane_id}: count={self.vehicle_count}, "
            f"speed={self.avg_speed:.1f}m/s, queue={self.queue_length}, "
            f"wait={self.waiting_time:.1f}s, emergency={self.has_emergency_vehicle})"
        )
