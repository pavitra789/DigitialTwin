"""
traffic_dataset.py
------------------
Handles feature extraction and dataset generation for short-horizon
lane-level traffic congestion prediction.

Features extracted per lane:
  - lane_index: Encoded numeric ID of lane
  - current_vehicle_count: Instantaneous vehicle count at step t
  - lag1_vehicle_count: Vehicle count at step t-1
  - lag2_vehicle_count: Vehicle count at step t-2
  - avg_speed: Current mean speed (m/s)
  - queue_length: Current halted vehicle count
  - waiting_time: Accumulated waiting time (s)
  - occupancy: Estimated lane occupancy percentage

Target:
  - next_interval_vehicle_count: Expected vehicle count at step t+1 (next 10-second interval)
"""

from collections import deque
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


LANE_TO_INDEX = {
    "north_in_0": 0,
    "south_in_0": 1,
    "east_in_0": 2,
    "west_in_0": 3,
}

FEATURE_COLUMNS = [
    "lane_index",
    "current_vehicle_count",
    "lag1_vehicle_count",
    "lag2_vehicle_count",
    "avg_speed",
    "queue_length",
    "waiting_time",
    "occupancy",
]


class FeatureExtractor:
    """Maintains a rolling window of lane metrics to extract lag features in real time."""

    def __init__(self, lane_ids: List[str], window_size: int = 3):
        self.lane_ids = lane_ids
        self.window_size = window_size
        # Store historical counts per lane: deque of past counts
        self.history: Dict[str, deque] = {
            lid: deque([0] * window_size, maxlen=window_size) for lid in lane_ids
        }

    def update_and_extract(self, lane_id: str, current_count: int, avg_speed: float, queue: int, wait_time: float, occupancy: float) -> np.ndarray:
        """
        Updates history with the latest vehicle count and returns the 1D feature array for live inference.
        """
        hist = self.history.get(lane_id)
        if hist is None:
            hist = deque([0] * self.window_size, maxlen=self.window_size)
            self.history[lane_id] = hist

        # Lag 1 and Lag 2 before pushing current count
        lag1 = hist[-1]
        lag2 = hist[-2] if len(hist) >= 2 else 0

        # Push current into history
        hist.append(current_count)

        lane_idx = LANE_TO_INDEX.get(lane_id, 0)
        feature_vector = np.array([
            lane_idx,
            current_count,
            lag1,
            lag2,
            avg_speed,
            queue,
            wait_time,
            occupancy
        ], dtype=float)

        return feature_vector.reshape(1, -1)


def generate_training_dataset(num_samples_per_lane: int = 400, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic training dataset modeling intersection dynamics:
    Simulates peak periods, arrival variance (Poisson), queue formation, and speed decay.
    """
    np.random.seed(random_seed)
    records = []

    lanes = list(LANE_TO_INDEX.keys())

    for lane_id in lanes:
        lane_idx = LANE_TO_INDEX[lane_id]
        
        # NS approaches typically have higher base traffic rate than EW
        base_arrival_rate = 6.0 if "north" in lane_id or "south" in lane_id else 3.5

        # Simulate a time series of steps
        current_count = int(np.random.poisson(base_arrival_rate))
        lag1 = int(np.random.poisson(base_arrival_rate))
        lag2 = int(np.random.poisson(base_arrival_rate))

        for step in range(num_samples_per_lane):
            # Dynamic surge factor (rush-hour wave simulation)
            rush_hour_mult = 1.0 + 0.6 * np.sin(step / 30.0)
            arrival = np.random.poisson(base_arrival_rate * rush_hour_mult)

            # Queuing and speed physics
            queue = int(max(0, current_count * 0.45 + np.random.normal(0, 1)))
            avg_speed = max(1.5, 13.89 - (current_count * 0.55) - (queue * 0.4) + np.random.normal(0, 0.5))
            wait_time = max(0.0, queue * 4.5 + np.random.exponential(2.0))
            occupancy = min(100.0, current_count * 5.0 + queue * 3.5)

            # Target: next interval count (t + 10s) based on momentum and dynamic inflow
            next_count = max(0, int(0.65 * current_count + 0.25 * arrival + 0.1 * lag1 + np.random.normal(0, 1.2)))

            records.append({
                "lane_id": lane_id,
                "lane_index": lane_idx,
                "current_vehicle_count": current_count,
                "lag1_vehicle_count": lag1,
                "lag2_vehicle_count": lag2,
                "avg_speed": round(avg_speed, 2),
                "queue_length": queue,
                "waiting_time": round(wait_time, 2),
                "occupancy": round(occupancy, 2),
                "next_interval_vehicle_count": next_count,
            })

            # Advance rolling window
            lag2 = lag1
            lag1 = current_count
            current_count = next_count

    df = pd.DataFrame(records)
    return df
