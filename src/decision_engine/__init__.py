"""
Decision Engine Module for Smart Traffic Management System.
Contains Adaptive Traffic Signal Control and Emergency Vehicle Priority logic.
"""

from .adaptive_signal import AdaptiveSignalController
from .emergency_priority import EmergencyPriorityController

__all__ = ["AdaptiveSignalController", "EmergencyPriorityController"]
