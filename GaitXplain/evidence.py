import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "joint_angles": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle"
    ],
    "left_right_asymmetry": [
        "hip_angle_asymmetry", "knee_angle_asymmetry",
        "thigh_length_asymmetry", "shank_length_asymmetry"
    ],
    "joint_angular_velocity": [
        "left_hip_angle_velocity", "right_hip_angle_velocity",
        "left_knee_angle_velocity", "right_knee_angle_velocity"
    ],
    "limb_segment_lengths": [
        "left_thigh_length", "right_thigh_length",
        "left_shank_length", "right_shank_length"
    ],
    "joint_linear_velocity": [
        "temporal_left_hip_x_velocity", "temporal_left_hip_y_velocity",
        "temporal_right_hip_x_velocity", "temporal_right_hip_y_velocity",
        "temporal_left_knee_x_velocity", "temporal_left_knee_y_velocity",
        "temporal_right_knee_x_velocity", "temporal_right_knee_y_velocity",
        "temporal_left_ankle_x_velocity", "temporal_left_ankle_y_velocity",
        "temporal_right_ankle_x_velocity", "temporal_right_ankle_y_velocity"
    ],
    "joint_acceleration": [
        "temporal_left_knee_acceleration_magnitude",
        "temporal_right_knee_acceleration_magnitude",
        "temporal_left_ankle_acceleration_magnitude",
        "temporal_right_ankle_acceleration_magnitude"
    ]
}


def build_biomechanical_evidence(report):
    summary = report.get("feature_summary")
    if not summary:
        return {
            "available": False,
            "reason": (
                "No frame-level feature summary found. Rerun prediction "
                "with the updated predict.py."
            )
        }

    all_features = summary.get("features", {})
    grouped = {}

    for group, names in FEATURE_GROUPS.items():
        grouped[group] = {
            name: all_features[name]
            for name in names
            if name in all_features
        }

    return {
        "available": True,
        "frames_summarised": summary.get("frames_summarised"),
        "feature_count": summary.get("feature_count"),
        "coordinate_note": summary.get("coordinate_note"),
        "feature_groups": grouped,
        "all_feature_statistics": all_features,
        "interpretation_note": (
            "These are descriptive statistics of engineered pose features. "
            "They are not clinical thresholds, diagnoses, or proof that a "
            "specific gait disorder is present."
        )
    }


def build_evidence(report_path):
    report_path = Path(report_path)
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    video = report["video_prediction"]
    windows = report.get("windows", [])
    if not windows:
        raise ValueError("No window-level predictions found in the report.")

    gait_predictions = [
        w.get("gait_prediction") for w in windows if w.get("gait_prediction")
    ]
    binary_predictions = [
        w.get("binary_prediction") for w in windows if w.get("binary_prediction")
    ]
    gait_counts = Counter(gait_predictions)
    binary_counts = Counter(binary_predictions)

    transitions = sum(
        gait_predictions[i] != gait_predictions[i - 1]
        for i in range(1, len(gait_predictions))
    )

    max_gait_probabilities = []
    for window in windows:
        probs = window.get("gait_probabilities", {})
        if probs:
            max_gait_probabilities.append(max(probs.values()))

    attention = [
        w.get("attention_weights", [])
        for w in windows if w.get("attention_weights")
    ]
    attention_summary = None
    if attention:
        attention_array = np.asarray(attention, dtype=float)
        mean_attention = attention_array.mean(axis=0)
        attention_summary = {
            "number_of_time_steps": int(len(mean_attention)),
            "highest_attention_time_step": int(np.argmax(mean_attention) + 1),
            "lowest_attention_time_step": int(np.argmin(mean_attention) + 1),
            "mean_attention_weights": mean_attention.round(6).tolist()
        }

    evidence = {
        "video_prediction": {
            "binary_class": video.get("binary_class"),
            "normal_probability": video.get("normal_probability"),
            "abnormal_probability": video.get("abnormal_probability"),
            "gait_class": video.get("gait_class"),
            "gait_probabilities": video.get("gait_probabilities")
        },
        "window_analysis": {
            "windows_evaluated": len(windows),
            "binary_prediction_counts": dict(binary_counts),
            "gait_prediction_counts": dict(gait_counts),
            "gait_class_transitions": int(transitions),
            "gait_transition_rate": round(
                transitions / max(len(gait_predictions) - 1, 1), 4
            ),
            "mean_maximum_gait_probability": (
                round(float(np.mean(max_gait_probabilities)), 4)
                if max_gait_probabilities else None
            )
        },
        "attention_analysis": attention_summary,
        "biomechanical_evidence": build_biomechanical_evidence(report),
        "feature_file": report.get("frame_level_features_csv"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "limitations": report.get("limitations", [])
    }
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = build_evidence(args.report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=4)

    print("Evidence extraction complete.")
    print(f"Windows analysed: {evidence['window_analysis']['windows_evaluated']}")
    print(f"Predicted gait class: {evidence['video_prediction']['gait_class']}")
    print(
        "Biomechanical evidence available: "
        f"{evidence['biomechanical_evidence']['available']}"
    )
    print(f"Evidence saved to: {output_path}")


if __name__ == "__main__":
    main()
