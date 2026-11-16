import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import TemporalCNNBiLSTM
from inference_features import FEATURES, extract_features, make_sequences

BINARY_LABELS = {0: "Normal", 1: "Abnormal"}
GAIT_LABELS = {
    0: "Ataxic", 1: "Choreiform", 2: "Diplegic",
    3: "Heimplegic", 4: "Myopathic", 5: "Neuropathic",
    6: "Parkinsonian"
}


def load_model(model_dir, device):
    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    scaler_path = model_dir / "results" / "feature_scaler.npz"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    if checkpoint["input_features"] != len(FEATURES):
        raise ValueError("Checkpoint feature count does not match extractor.")

    if list(checkpoint["feature_names"]) != FEATURES:
        raise ValueError("Checkpoint feature order does not match extractor.")

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    if list(scaler["feature_names"]) != FEATURES:
        raise ValueError("Scaler feature order does not match extractor.")

    model = TemporalCNNBiLSTM(
        input_features=len(FEATURES),
        num_gait_classes=len(GAIT_LABELS)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, mean, std, checkpoint


def build_feature_summary(feature_df):
    """Summarise unscaled frame-level features; units remain feature-dependent."""
    summary = {}
    for feature in FEATURES:
        values = pd.to_numeric(feature_df[feature], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()

        summary[feature] = {
            "valid_frames": int(finite.size),
            "missing_frames": int(len(values) - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std(ddof=0)) if finite.size else None,
            "median": float(finite.median()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None
        }

    return {
        "frames_summarised": int(len(feature_df)),
        "feature_count": len(FEATURES),
        "coordinate_note": (
            "Feature values are the original extracted values before model "
            "scaling. Values derived from image coordinates are not calibrated "
            "physical measurements; interpret units according to each feature."
        ),
        "features": summary
    }


def predict(keypoints_csv, output_dir, width, height, batch_size=64):
    project_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, mean, std, checkpoint = load_model(project_dir, device)
    print(f"Loaded checkpoint from epoch: {checkpoint.get('best_epoch', 'unknown')}")

    print("\nExtracting features...")
    feature_df = extract_features(keypoints_csv, width=width, height=height)
    sequences, frame_ranges = make_sequences(feature_df)

    print(f"Frames: {len(feature_df)}")
    print(f"Sequences: {len(sequences)}")

    # Preserve the original, unscaled features for evidence/reporting.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "frame_level_features.csv"
    feature_df.to_csv(features_path, index=False)
    feature_summary = build_feature_summary(feature_df)

    # The classifier continues to receive the exact same scaled sequences.
    sequences = (sequences - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    sequences = np.clip(sequences, -8.0, 8.0).astype(np.float32)

    all_binary_probs, all_gait_probs, all_attention = [], [], []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(
                sequences[start:start + batch_size]
            ).to(device)
            outputs = model(batch)
            all_binary_probs.append(
                torch.softmax(outputs["binary"], dim=1).cpu().numpy()
            )
            all_gait_probs.append(
                torch.softmax(outputs["gait"], dim=1).cpu().numpy()
            )
            all_attention.append(outputs["attention"].cpu().numpy())

    binary_probs = np.concatenate(all_binary_probs)
    gait_probs = np.concatenate(all_gait_probs)
    attention = np.concatenate(all_attention)

    binary_predictions = binary_probs.argmax(axis=1)
    gait_predictions = gait_probs.argmax(axis=1)

    mean_binary_probs = binary_probs.mean(axis=0)
    video_binary = int(mean_binary_probs.argmax())

    abnormal_indices = np.where(binary_predictions == 1)[0]
    if len(abnormal_indices) == 0:
        abnormal_indices = np.array([int(np.argmax(binary_probs[:, 1]))])

    mean_gait_probs = gait_probs[abnormal_indices].mean(axis=0)
    video_gait = int(mean_gait_probs.argmax())

    windows = []
    for i in range(len(sequences)):
        windows.append({
            "window": i + 1,
            "start_frame": frame_ranges[i][0],
            "end_frame": frame_ranges[i][1],
            "binary_prediction": BINARY_LABELS[int(binary_predictions[i])],
            "normal_probability": float(binary_probs[i, 0]),
            "abnormal_probability": float(binary_probs[i, 1]),
            "gait_prediction": GAIT_LABELS[int(gait_predictions[i])],
            "gait_probabilities": {
                GAIT_LABELS[j]: float(gait_probs[i, j])
                for j in range(len(GAIT_LABELS))
            },
            "attention_weights": attention[i].tolist()
        })

    report = {
        "input_keypoints_csv": str(Path(keypoints_csv).resolve()),
        "frame_level_features_csv": str(features_path.resolve()),
        "feature_summary": feature_summary,
        "frames_processed": len(feature_df),
        "sequences_evaluated": len(sequences),
        "sequence_length": 32,
        "stride": 16,
        "checkpoint_epoch": int(checkpoint.get("best_epoch", -1)),
        "device": str(device),
        "video_prediction": {
            "binary_class": BINARY_LABELS[video_binary],
            "normal_probability": float(mean_binary_probs[0]),
            "abnormal_probability": float(mean_binary_probs[1]),
            "gait_class": GAIT_LABELS[video_gait] if video_binary == 1 else None,
            "gait_probabilities": {
                GAIT_LABELS[j]: float(mean_gait_probs[j])
                for j in range(len(GAIT_LABELS))
            } if video_binary == 1 else None
        },
        "windows": windows,
        "limitations": [
            "Predictions are model outputs, not clinical diagnoses.",
            "Softmax probabilities are not calibrated diagnostic confidence.",
            "The model has been trained on nine recordings; independent-subject generalization has not been established.",
            "Frame-level feature summaries are descriptive and are not validated clinical gait measurements.",
            "Image-coordinate-derived values are not physical distances or speeds without camera calibration."
        ]
    }

    report_path = output_dir / "prediction_report.json"
    windows_path = output_dir / "window_predictions.csv"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    pd.DataFrame([
        {
            "window": w["window"],
            "start_frame": w["start_frame"],
            "end_frame": w["end_frame"],
            "binary_prediction": w["binary_prediction"],
            "normal_probability": w["normal_probability"],
            "abnormal_probability": w["abnormal_probability"],
            "gait_prediction": w["gait_prediction"]
        }
        for w in windows
    ]).to_csv(windows_path, index=False)

    print("\n" + "=" * 55)
    print("GAITXPLAIN PREDICTION")
    print("=" * 55)
    print(f"Binary result: {BINARY_LABELS[video_binary]}")
    print(f"Normal probability:   {mean_binary_probs[0]:.4f}")
    print(f"Abnormal probability: {mean_binary_probs[1]:.4f}")
    if video_binary == 1:
        print(f"Predicted gait category: {GAIT_LABELS[video_gait]}")
        print(f"Gait probability: {mean_gait_probs[video_gait]:.4f}")
    print(f"\nFrame-level features saved: {features_path}")
    print(f"Report saved: {report_path}")
    print(f"Window predictions saved: {windows_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GaitXplain two-level gait prediction"
    )
    parser.add_argument("--input", required=True, help="YOLO keypoints CSV")
    parser.add_argument("--output", default="results/predictions")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    predict(args.input, args.output, args.width, args.height, args.batch_size)


if __name__ == "__main__":
    main()
