"""
AI Prediction Module for Smart Traffic Management System.
Provides short-horizon lane-level traffic congestion prediction using scikit-learn.
"""

from .congestion_predictor import CongestionPredictor
from .traffic_dataset import generate_training_dataset, FeatureExtractor

__all__ = ["CongestionPredictor", "generate_training_dataset", "FeatureExtractor"]
