"""
congestion_predictor.py
-----------------------
AI-Driven Traffic Congestion Predictor using scikit-learn.
Employs Random Forest / Gradient Boosting regression to forecast next-interval
traffic volume per lane and assigns categorical congestion severity levels.
"""

import os
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .traffic_dataset import FEATURE_COLUMNS, FeatureExtractor, generate_training_dataset


class CongestionPredictor:
    """
    Scikit-learn based short-horizon congestion forecaster.
    """

    CONGESTION_LEVELS = {
        "LOW": (0, 4),        # Smooth flow
        "MEDIUM": (5, 9),     # Moderate density
        "HIGH": (10, 14),     # Heavy congestion — candidate for rerouting / extended green
        "CRITICAL": (15, 999) # Severe gridlock — urgent intervention required
    }

    def __init__(self, model_type: str = "rf", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "congestion_model.joblib"
        )
        self.model = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.metrics: Dict[str, float] = {}

    def init_feature_extractor(self, lane_ids: list):
        """Initializes the rolling history feature extractor."""
        self.feature_extractor = FeatureExtractor(lane_ids)

    def classify_congestion(self, predicted_count: float) -> str:
        """Categorizes predicted count into a human-interpretable congestion level."""
        count = max(0.0, round(predicted_count))
        for level, (low, high) in self.CONGESTION_LEVELS.items():
            if low <= count <= high:
                return level
        return "CRITICAL"

    def train(self, dataset_size: int = 500, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Trains the ML model on synthetic/historical intersection telemetry.
        """
        print(f"[AI MODEL] Generating training dataset ({dataset_size} steps/lane)...")
        df = generate_training_dataset(num_samples_per_lane=dataset_size, random_seed=random_state)

        X = df[FEATURE_COLUMNS].values
        y = df["next_interval_vehicle_count"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if self.model_type == "gbr":
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1
            )

        print(f"[AI MODEL] Fitting {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        self.metrics = {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "test_samples": len(y_test),
        }

        print(f"[AI MODEL] Training Complete -> R2: {self.metrics['r2_score']}, MAE: {self.metrics['mae']}, RMSE: {self.metrics['rmse']}")
        self.save_model()
        return self.metrics

    def save_model(self):
        """Serializes trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"model": self.model, "metrics": self.metrics, "model_type": self.model_type}, self.model_path)
        print(f"[AI MODEL] Saved model artifact to: {self.model_path}")

    def load_model(self) -> bool:
        """Loads trained model from disk if available."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.metrics = data.get("metrics", {})
                self.model_type = data.get("model_type", self.model_type)
                print(f"[AI MODEL] Successfully loaded saved model from: {self.model_path}")
                return True
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")
        return False

    def predict_lane(self, lane_id: str, current_count: int, avg_speed: float, queue: int, wait_time: float, occupancy: float) -> Tuple[float, str]:
        """
        Performs live short-horizon prediction for a given lane.

        Returns:
            (predicted_vehicle_count: float, congestion_level: str)
        """
        if self.model is None:
            if not self.load_model():
                print("[AI MODEL] Model not found on disk. Auto-training now...")
                self.train()

        if self.feature_extractor is None:
            self.init_feature_extractor([lane_id])

        features = self.feature_extractor.update_and_extract(
            lane_id=lane_id,
            current_count=current_count,
            avg_speed=avg_speed,
            queue=queue,
            wait_time=wait_time,
            occupancy=occupancy,
        )

        pred_count = float(self.model.predict(features)[0])
        pred_count = max(0.0, round(pred_count, 1))
        level = self.classify_congestion(pred_count)

        return pred_count, level
