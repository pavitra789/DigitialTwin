"""
test_simulation.py
------------------
Automated test suite for the Digital Twin Smart Traffic Management System.
Verifies:
  1. Digital Twin state modeling & serialization.
  2. Adaptive Green Time formula scaling and boundary enforcement [MIN_GREEN, MAX_GREEN].
  3. Emergency Vehicle Priority preemption trigger and state override.
  4. Snapshot generation for API/WebSocket consumers.
"""

import sys
import os
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.digital_twin.lane_twin import LaneTwin
from src.digital_twin.vehicle_twin import VehicleTwin
from src.digital_twin.signal_twin import SignalTwin
from src.digital_twin.digital_twin_system import DigitalTwin
from src.decision_engine.adaptive_signal import AdaptiveSignalController
from src.decision_engine.emergency_priority import EmergencyPriorityController


class TestDigitalTwinSystem(unittest.TestCase):

    def setUp(self):
        self.lane_ids = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
        self.phase_map = {
            "north_in_0": 0,
            "south_in_0": 0,
            "east_in_0": 2,
            "west_in_0": 2,
        }
        self.twin = DigitalTwin(self.lane_ids, traffic_light_id="TL1")
        self.adaptive_controller = AdaptiveSignalController(
            traffic_light_id="TL1",
            phase_map=self.phase_map,
            min_green=15,
            max_green=60,
            base_green=25
        )
        self.emergency_controller = EmergencyPriorityController(
            traffic_light_id="TL1",
            phase_map=self.phase_map,
            emergency_green_time=55
        )

    def test_vehicle_twin_state(self):
        v = VehicleTwin("veh_01", vehicle_type="passenger")
        v.update(speed=12.5, position=(1.6, 120.0), lane_id="north_in_0", distance=80.0, waiting_time=0.0)
        
        self.assertEqual(v.vehicle_id, "veh_01")
        self.assertEqual(v.speed, 12.5)
        self.assertFalse(v.is_emergency)
        
        data = v.to_dict()
        self.assertEqual(data["vehicle_id"], "veh_01")
        self.assertEqual(data["is_emergency"], False)

    def test_emergency_vehicle_twin(self):
        v_em = VehicleTwin("ambulance_01", vehicle_type="emergency")
        v_em.update(speed=18.0, position=(1.6, 50.0), lane_id="north_in_0", distance=150.0, waiting_time=2.0)
        self.assertTrue(v_em.is_emergency)

    def test_adaptive_green_calculation(self):
        # Scenario 1: Low traffic -> should give base green
        self.twin.lanes["north_in_0"].update_mock(vehicle_count=2, avg_speed=13.0, queue_length=0, waiting_time=0.0)
        time_low = self.adaptive_controller.compute_green_time(self.twin, "north_in_0")
        self.assertGreaterEqual(time_low, 25)

        # Scenario 2: High traffic & heavy queue -> should scale up towards max_green
        self.twin.lanes["south_in_0"].update_mock(vehicle_count=20, avg_speed=2.0, queue_length=12, waiting_time=95.0)
        time_high = self.adaptive_controller.compute_green_time(self.twin, "south_in_0")
        self.assertEqual(time_high, 60)  # Capped at max_green (60s)

    def test_emergency_priority_override(self):
        # No emergency vehicle initially
        self.twin.lanes["north_in_0"].update_mock(vehicle_count=10, avg_speed=5.0, queue_length=4, waiting_time=10.0, has_emergency=False)
        is_em, lane, phase = self.emergency_controller.check_and_apply_priority(self.twin)
        self.assertFalse(is_em)

        # Emergency vehicle enters east_in_0
        self.twin.lanes["east_in_0"].update_mock(vehicle_count=5, avg_speed=8.0, queue_length=1, waiting_time=0.0, has_emergency=True)
        is_em, lane, phase = self.emergency_controller.check_and_apply_priority(self.twin)
        
        self.assertTrue(is_em)
        self.assertEqual(lane, "east_in_0")
        self.assertEqual(phase, 2)  # Phase 2 is East-West Green
        self.assertTrue(self.twin.signal.is_overridden)

    def test_snapshot_format(self):
        self.twin.lanes["north_in_0"].update_mock(vehicle_count=6, avg_speed=10.0, queue_length=2, waiting_time=5.0)
        snapshot = self.twin.get_state_snapshot()
        
        self.assertIn("summary", snapshot)
        self.assertIn("lanes", snapshot)
        self.assertIn("signal", snapshot)
        self.assertEqual(snapshot["lanes"]["north_in_0"]["vehicle_count"], 6)


if __name__ == "__main__":
    unittest.main()
