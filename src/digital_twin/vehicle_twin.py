"""
vehicle_twin.py
---------------
Virtual representation of an individual vehicle in the Digital Twin.
Mirrors telemetry data such as position, speed, type, and waiting time.
"""

class VehicleTwin:
    """Represents a virtual twin of an individual vehicle."""

    def __init__(self, vehicle_id: str, vehicle_type: str = "passenger"):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.speed = 0.0          # Current speed in m/s
        self.position = (0.0, 0.0)# (x, y) coordinates in simulation
        self.lane_id = ""         # Current lane ID
        self.distance = 0.0       # Distance along the lane in meters
        self.waiting_time = 0.0   # Accumulated waiting/idling time in seconds
        self.is_emergency = (vehicle_type == "emergency")

    def update(self, speed: float, position: tuple, lane_id: str, distance: float, waiting_time: float):
        """Update live telemetry attributes for this vehicle."""
        self.speed = speed
        self.position = position
        self.lane_id = lane_id
        self.distance = distance
        self.waiting_time = waiting_time

    def to_dict(self) -> dict:
        """Serializes vehicle state to a dictionary for API/WebSocket/logging."""
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "speed": round(self.speed, 2),
            "position": {"x": round(self.position[0], 2), "y": round(self.position[1], 2)},
            "lane_id": self.lane_id,
            "distance": round(self.distance, 2),
            "waiting_time": round(self.waiting_time, 2),
            "is_emergency": self.is_emergency,
        }

    def __repr__(self) -> str:
        return (
            f"VehicleTwin(id={self.vehicle_id}, type={self.vehicle_type}, "
            f"lane={self.lane_id}, speed={self.speed:.1f}m/s, wait={self.waiting_time:.1f}s)"
        )
