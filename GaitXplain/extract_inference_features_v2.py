import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CONF_THRESHOLD = 0.50
MAX_GAP = 10
WINDOW = 15
POLYORDER = 2

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


def interpolate_and_smooth(series):
    s = series.astype(float).copy()
    s = s.interpolate(method="linear", limit=MAX_GAP, limit_area="inside")

    values = s.to_numpy(copy=True)
    valid = np.isfinite(values)
    start = 0

    while start < len(values):
        if not valid[start]:
            start += 1
            continue

        end = start
        while end < len(values) and valid[end]:
            end += 1

        segment = values[start:end]
        if len(segment) >= WINDOW:
            values[start:end] = savgol_filter(
                segment, WINDOW, POLYORDER
            )

        start = end

    return pd.Series(values, index=series.index)


def angle(a, b, c):
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)

    cosine = np.divide(
        np.sum(ba * bc, axis=1),
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 1e-8
    )

    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def distance(a, b):
    return np.linalg.norm(a - b, axis=1)


def add_speed(df, x_col, y_col, prefix, fps):
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    dx = np.gradient(x) * fps
    dy = np.gradient(y) * fps

    df[f"{prefix}_velocity_x"] = dx
    df[f"{prefix}_velocity_y"] = dy
    df[f"{prefix}_speed"] = np.sqrt(dx**2 + dy**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("FPS and video dimensions must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    name = input_path.stem
    if name.endswith("_keypoints"):
        name = name[:-len("_keypoints")]

    output_path = output_dir / f"{name}_gait_features.csv"
    summary_path = output_dir / f"{name}_gait_summary.csv"

    df = pd.read_csv(input_path)
    n = len(df)

    required = ["frame", "person_detected"]
    for joint in JOINTS:
        required.extend([
            f"{joint}_x", f"{joint}_y", f"{joint}_confidence"
        ])

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required keypoint columns: {missing}")

    if n < 2:
        raise ValueError("At least two video frames are required.")

    for joint in JOINTS:
        x, y, conf = (
            f"{joint}_x",
            f"{joint}_y",
            f"{joint}_confidence"
        )

        valid = (
            (df["person_detected"] == 1) &
            (df[conf] >= CONF_THRESHOLD)
        )

        df.loc[~valid, [x, y]] = np.nan
        df[x] = interpolate_and_smooth(df[x])
        df[y] = interpolate_and_smooth(df[y])

    def points(joint):
        return df[[f"{joint}_x", f"{joint}_y"]].to_numpy(dtype=float)

    lh, rh = points("left_hip"), points("right_hip")
    lk, rk = points("left_knee"), points("right_knee")
    la, ra = points("left_ankle"), points("right_ankle")
    ls, rs = points("left_shoulder"), points("right_shoulder")

    df["time_seconds"] = df["frame"] / args.fps

    # Pelvis midpoint, with a one-hip fallback when needed.
    pelvis = np.full((n, 2), np.nan)
    both_hips = np.isfinite(lh).all(axis=1) & np.isfinite(rh).all(axis=1)
    only_left = np.isfinite(lh).all(axis=1) & ~np.isfinite(rh).all(axis=1)
    only_right = np.isfinite(rh).all(axis=1) & ~np.isfinite(lh).all(axis=1)

    pelvis[both_hips] = (lh[both_hips] + rh[both_hips]) / 2
    pelvis[only_left] = lh[only_left]
    pelvis[only_right] = rh[only_right]

    df["pelvis_x"] = pelvis[:, 0]
    df["pelvis_y"] = pelvis[:, 1]

    df["shoulder_width_px"] = distance(ls, rs)
    df["hip_width_px"] = distance(lh, rh)

    df["left_thigh_length_px"] = distance(lh, lk)
    df["right_thigh_length_px"] = distance(rh, rk)
    df["left_shank_length_px"] = distance(lk, la)
    df["right_shank_length_px"] = distance(rk, ra)

    # Projected 2D joint angles, not calibrated clinical measurements.
    df["left_hip_angle_deg"] = angle(ls, lh, lk)
    df["right_hip_angle_deg"] = angle(rs, rh, rk)
    df["left_knee_angle_deg"] = angle(lh, lk, la)
    df["right_knee_angle_deg"] = angle(rh, rk, ra)

    df["hip_angle_asymmetry_deg"] = (
        df["left_hip_angle_deg"] - df["right_hip_angle_deg"]
    ).abs()

    df["knee_angle_asymmetry_deg"] = (
        df["left_knee_angle_deg"] - df["right_knee_angle_deg"]
    ).abs()

    df["pelvis_x_normalized"] = df["pelvis_x"] / args.width
    df["pelvis_y_normalized"] = df["pelvis_y"] / args.height

    shoulder_midpoint = (ls + rs) / 2
    torso = distance(shoulder_midpoint, pelvis)
    torso = pd.Series(torso).replace(0, np.nan).to_numpy()

    df["pelvis_x_body_relative"] = (
        df["pelvis_x"] - shoulder_midpoint[:, 0]
    ) / torso

    df["pelvis_y_body_relative"] = (
        df["pelvis_y"] - shoulder_midpoint[:, 1]
    ) / torso

    add_speed(df, "pelvis_x", "pelvis_y", "pelvis", args.fps)

    feature_cols = [
        "frame", "time_seconds", "person_detected",
        "pelvis_x", "pelvis_y",
        "pelvis_x_normalized", "pelvis_y_normalized",
        "pelvis_x_body_relative", "pelvis_y_body_relative",
        "pelvis_velocity_x", "pelvis_velocity_y", "pelvis_speed",
        "shoulder_width_px", "hip_width_px",
        "left_thigh_length_px", "right_thigh_length_px",
        "left_shank_length_px", "right_shank_length_px",
        "left_hip_angle_deg", "right_hip_angle_deg",
        "left_knee_angle_deg", "right_knee_angle_deg",
        "hip_angle_asymmetry_deg", "knee_angle_asymmetry_deg"
    ]

    features = df[feature_cols].copy()
    features.to_csv(output_path, index=False)

    numeric = features.select_dtypes(include="number")

    summary = pd.DataFrame({
        "feature": numeric.columns,
        "mean": numeric.mean().values,
        "std": numeric.std().values,
        "min": numeric.min().values,
        "median": numeric.median().values,
        "max": numeric.max().values,
        "valid_frames": numeric.count().values
    })

    summary.to_csv(summary_path, index=False)

    print("\n===== GAIT FEATURE EXTRACTION RESULT =====")
    print("Input:", input_path)
    print("Frames:", n)
    print("FPS:", args.fps)
    print("Resolution:", args.width, "x", args.height)
    print("Feature columns:", len(feature_cols))
    print("Feature output:", output_path)
    print("Summary output:", summary_path)

    for column in [
        "pelvis_speed",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "knee_angle_asymmetry_deg"
    ]:
        print(
            f"{column}: mean={features[column].mean():.4f}, "
            f"valid={features[column].count()}/{n}"
        )


if __name__ == "__main__":
    main()
