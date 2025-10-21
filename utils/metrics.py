"""
Metrics utilities for evaluation.
"""
import numpy as np

def compute_matching_score(predictions, ground_truth):
    """
    Compute matching score between predictions and ground truth.
    This is a placeholder function - implement according to your specific needs.
    """

    return (predictions == ground_truth).mean()
