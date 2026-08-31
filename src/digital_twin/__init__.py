"""
Digital Twin Module for Smart Traffic Management System.
Contains virtual representations of vehicles, lanes, and traffic signals.
"""

from .vehicle_twin import VehicleTwin
from .lane_twin import LaneTwin
from .signal_twin import SignalTwin
from .digital_twin_system import DigitalTwin

__all__ = ["VehicleTwin", "LaneTwin", "SignalTwin", "DigitalTwin"]
