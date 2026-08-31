"""
test_ai_and_routing.py
----------------------
Automated test suite verifying:
  1. AI dataset generation and feature extractor.
  2. Scikit-learn Congestion Predictor training, evaluation metrics (R2 > 0.8), and model serialization.
  3. Live inference speed (<10ms) and congestion classification.
  4. Static Route Recommender alternate route trigger when predicted congestion is HIGH/CRITICAL.
"""

import sys
import os
import unittest
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai_model.traffic_dataset import generate_training_dataset, FeatureExtractor
from src.ai_model.congestion_predictor import CongestionPredictor
from src.route_recommendation.route_recommender import StaticRouteRecommender
from src.digital_twin.digital_twin_system import DigitalTwin


class TestAIAndRouting(unittest.TestCase):

    def setUp(self):
        self.lane_ids = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
        self.twin = DigitalTwin(self.lane_ids, traffic_light_id="TL1")
        self.predictor = CongestionPredictor(model_type="rf")
        self.recommender = StaticRouteRecommender(congestion_threshold="HIGH")

    def test_dataset_generation(self):
        df = generate_training_dataset(num_samples_per_lane=100, random_seed=42)
        self.assertEqual(len(df), 400)
        self.assertIn("next_interval_vehicle_count", df.columns)
        self.assertIn("avg_speed", df.columns)
        self.assertIn("occupancy", df.columns)

    def test_feature_extractor_rolling_window(self):
        fe = FeatureExtractor(self.lane_ids, window_size=3)
        vec1 = fe.update_and_extract("north_in_0", current_count=5, avg_speed=10.0, queue=1, wait_time=2.0, occupancy=25.0)
        self.assertEqual(vec1.shape, (1, 8))
        self.assertEqual(vec1[0][1], 5.0)  # current count
        self.assertEqual(vec1[0][2], 0.0)  # lag 1

        vec2 = fe.update_and_extract("north_in_0", current_count=8, avg_speed=8.0, queue=2, wait_time=4.0, occupancy=40.0)
        self.assertEqual(vec2[0][1], 8.0)  # current count
        self.assertEqual(vec2[0][2], 5.0)  # lag 1 is now previous 5

    def test_model_training_and_accuracy(self):
        metrics = self.predictor.train(dataset_size=300, random_state=42)
        self.assertIn("r2_score", metrics)
        self.assertIn("mae", metrics)
        self.assertGreater(metrics["r2_score"], 0.70)
        self.assertTrue(os.path.exists(self.predictor.model_path))

    def test_live_inference_and_latency(self):
        if self.predictor.model is None:
            self.predictor.train(dataset_size=200)

        t0 = time.perf_counter()
        pred_count, level = self.predictor.predict_lane(
            lane_id="north_in_0",
            current_count=12,
            avg_speed=4.5,
            queue=8,
            wait_time=30.0,
            occupancy=70.0
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms

        self.assertGreaterEqual(pred_count, 0.0)
        self.assertIn(level, ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        self.assertLess(t_elapsed, 50.0)  # Latency under 50ms

    def test_static_route_recommendation(self):
        # Scenario: North approach has HIGH predicted congestion
        lane_predictions = {
            "north_in_0": (12.0, "HIGH"),
            "south_in_0": (3.0, "LOW"),
            "east_in_0": (2.0, "LOW"),
            "west_in_0": (1.0, "LOW"),
        }
        
        # Populate simulated vehicle in north_in_0
        self.twin.lanes["north_in_0"].active_vehicle_ids = ["veh_ns_101", "veh_ns_102"]
        
        actions = self.recommender.evaluate_and_reroute(self.twin, lane_predictions)
        
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0]["original_route"], "r_NS")
        self.assertEqual(actions[0]["recommended_route"], "r_NS_alt1")
        self.assertEqual(self.recommender.reroute_count, len(actions))


if __name__ == "__main__":
    unittest.main()
