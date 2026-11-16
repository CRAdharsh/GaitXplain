import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 30.0
CONFIDENCE_THRESHOLD = 0.50
MAX_INTERPOLATION_GAP = 10
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
TEMPORAL_SMOOTHING_WINDOW = 5
SEQUENCE_LENGTH = 32
STRIDE = 16

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

FEATURES = [
    "left_hip_angle", "right_hip_angle",
    "left_knee_angle", "right_knee_angle",
    "hip_angle_asymmetry", "knee_angle_asymmetry",
    "left_hip_angle_velocity", "right_hip_angle_velocity",
    "left_knee_angle_velocity", "right_knee_angle_velocity",
    "left_thigh_length", "right_thigh_length",
    "left_shank_length", "right_shank_length",
    "thigh_length_asymmetry", "shank_length_asymmetry",
    "temporal_left_hip_x_velocity",
    "temporal_left_hip_y_velocity",
    "temporal_right_hip_x_velocity",
    "temporal_right_hip_y_velocity",
    "temporal_left_knee_x_velocity",
    "temporal_left_knee_y_velocity",
    "temporal_right_knee_x_velocity",
    "temporal_right_knee_y_velocity",
    "temporal_left_ankle_x_velocity",
    "temporal_left_ankle_y_velocity",
    "temporal_right_ankle_x_velocity",
    "temporal_right_ankle_y_velocity",
    "temporal_left_knee_acceleration_magnitude",
    "temporal_right_knee_acceleration_magnitude",
    "temporal_left_ankle_acceleration_magnitude",
    "temporal_right_ankle_acceleration_magnitude"
]


def interpolate_short_gaps(series, max_gap=MAX_INTERPOLATION_GAP):
    values = np.asarray(series, dtype=float).copy()
    missing = ~np.isfinite(values)
    interpolated = np.zeros(len(values), dtype=bool)

    i = 0
    while i < len(values):
        if not missing[i]:
            i += 1
            continue

        start = i
        while i < len(values) and missing[i]:
            i += 1

        end = i
        gap_length = end - start

        if (
            gap_length <= max_gap
            and start > 0
            and end < len(values)
            and np.isfinite(values[start - 1])
            and np.isfinite(values[end])
        ):
            values[start:end] = np.linspace(
                values[start - 1],
                values[end],
                gap_length + 2
            )[1:-1]
            interpolated[start:end] = True

    return pd.Series(values), interpolated


def smooth_valid_sections(series):
    values = np.asarray(series, dtype=float).copy()
    valid = np.isfinite(values)

    i = 0
    while i < len(values):
        if not valid[i]:
            i += 1
            continue

        start = i
        while i < len(values) and valid[i]:
            i += 1

        end = i
        section_length = end - start
        window = min(SAVGOL_WINDOW, section_length)

        if window % 2 == 0:
            window -= 1

        if window >= 5:
            values[start:end] = savgol_filter(
                values[start:end],
                window_length=window,
                polyorder=min(SAVGOL_POLYORDER, window - 2),
                mode="interp"
            )

    return pd.Series(values)


def derivatives_without_gaps(series):
    values = np.asarray(series, dtype=float)
    velocity = np.full(len(values), np.nan)
    acceleration = np.full(len(values), np.nan)
    valid = np.isfinite(values)

    i = 0
    while i < len(values):
        if not valid[i]:
            i += 1
            continue

        start = i
        while i < len(values) and valid[i]:
            i += 1

        end = i
        if end - start >= 3:
            segment = values[start:end]
            vel = np.gradient(segment, 1.0 / FPS)
            acc = np.gradient(vel, 1.0 / FPS)
            velocity[start:end] = vel
            acceleration[start:end] = acc

    return velocity, acceleration


def calculate_angle(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if not (np.isfinite(a).all() and np.isfinite(b).all()
            and np.isfinite(c).all()):
        return np.nan

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-10:
        return np.nan

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def distance(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def build_confidence_coordinates(raw, width, height):
    result = pd.DataFrame({
        "frame": raw["frame"].to_numpy()
    })

    detected = (
        pd.to_numeric(raw["person_detected"], errors="coerce")
        .fillna(0).to_numpy() == 1
    )

    for joint in JOINTS:
        conf = pd.to_numeric(
            raw[f"{joint}_confidence"], errors="coerce"
        ).to_numpy(dtype=float)

        x = pd.to_numeric(
            raw[f"{joint}_x"], errors="coerce"
        ).to_numpy(dtype=float, copy=True)

        y = pd.to_numeric(
            raw[f"{joint}_y"], errors="coerce"
        ).to_numpy(dtype=float, copy=True)

        valid = (
            (conf >= CONFIDENCE_THRESHOLD)
            & detected
            & np.isfinite(x)
            & np.isfinite(y)
        )

        x[~valid] = np.nan
        y[~valid] = np.nan

        x, _ = interpolate_short_gaps(pd.Series(x))
        y, _ = interpolate_short_gaps(pd.Series(y))

        x = smooth_valid_sections(x).to_numpy()
        y = smooth_valid_sections(y).to_numpy()

        result[f"{joint}_image_x"] = x / float(width)
        result[f"{joint}_image_y"] = y / float(height)

    lhx = result["left_hip_image_x"].to_numpy()
    lhy = result["left_hip_image_y"].to_numpy()
    rhx = result["right_hip_image_x"].to_numpy()
    rhy = result["right_hip_image_y"].to_numpy()

    pelvis_valid = (
        np.isfinite(lhx) & np.isfinite(lhy)
        & np.isfinite(rhx) & np.isfinite(rhy)
    )

    pelvis_x = np.full(len(raw), np.nan)
    pelvis_y = np.full(len(raw), np.nan)
    pelvis_x[pelvis_valid] = (lhx[pelvis_valid] + rhx[pelvis_valid]) / 2
    pelvis_y[pelvis_valid] = (lhy[pelvis_valid] + rhy[pelvis_valid]) / 2

    result["pelvis_image_x"] = pelvis_x
    result["pelvis_image_y"] = pelvis_y

    for axis, values in [
        ("x", pelvis_x),
        ("y", pelvis_y)
    ]:
        velocity, acceleration = derivatives_without_gaps(values)
        result[f"pelvis_velocity_{axis}"] = velocity
        result[f"pelvis_acceleration_{axis}"] = acceleration

    result["pelvis_speed"] = np.hypot(
        result["pelvis_velocity_x"],
        result["pelvis_velocity_y"]
    )
    result["pelvis_acceleration_magnitude"] = np.hypot(
        result["pelvis_acceleration_x"],
        result["pelvis_acceleration_y"]
    )

    return result


def build_lower_body_features(coords):
    n = len(coords)
    out = pd.DataFrame({"frame": coords["frame"].to_numpy()})

    def point(joint):
        return np.column_stack([
            coords[f"{joint}_image_x"].to_numpy(dtype=float),
            coords[f"{joint}_image_y"].to_numpy(dtype=float)
        ])

    points = {
        joint: point(joint)
        for joint in [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
    }

    for joint, values in points.items():
        out[f"{joint}_x"] = values[:, 0]
        out[f"{joint}_y"] = values[:, 1]

    out["pelvis_x"] = (
        points["left_hip"][:, 0] + points["right_hip"][:, 0]
    ) / 2
    out["pelvis_y"] = (
        points["left_hip"][:, 1] + points["right_hip"][:, 1]
    ) / 2

    angle_specs = {
        "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle")
    }

    for name, (a, b, c) in angle_specs.items():
        out[name] = [
            calculate_angle(points[a][i], points[b][i], points[c][i])
            for i in range(n)
        ]

    out["hip_angle_asymmetry"] = (
        out["left_hip_angle"] - out["right_hip_angle"]
    ).abs()
    out["knee_angle_asymmetry"] = (
        out["left_knee_angle"] - out["right_knee_angle"]
    ).abs()

    for side in ["left", "right"]:
        out[f"{side}_thigh_length"] = [
            distance(points[f"{side}_hip"][i], points[f"{side}_knee"][i])
            for i in range(n)
        ]
        out[f"{side}_shank_length"] = [
            distance(points[f"{side}_knee"][i], points[f"{side}_ankle"][i])
            for i in range(n)
        ]

    out["thigh_length_asymmetry"] = (
        out["left_thigh_length"] - out["right_thigh_length"]
    ).abs()
    out["shank_length_asymmetry"] = (
        out["left_shank_length"] - out["right_shank_length"]
    ).abs()

    for joint in [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]:
        for axis in ["x", "y"]:
            vel, _ = derivatives_without_gaps(
                out[f"{joint}_{axis}"]
            )
            out[f"{joint}_velocity_{axis}"] = vel

        out[f"{joint}_speed"] = np.hypot(
            out[f"{joint}_velocity_x"],
            out[f"{joint}_velocity_y"]
        )

    for name in [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle"
    ]:
        vel, _ = derivatives_without_gaps(out[name])
        out[f"{name.replace('_angle', '')}_angle_velocity"] = vel

    for axis in ["x", "y"]:
        vel, _ = derivatives_without_gaps(out[f"pelvis_{axis}"])
        out[f"pelvis_velocity_{axis}"] = vel

    out["pelvis_speed"] = np.hypot(
        out["pelvis_velocity_x"],
        out["pelvis_velocity_y"]
    )

    return out


def prepare_temporal_coordinates(raw):
    df = raw.copy()
    joints = [
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]

    prepared = {}

    detected = (
        pd.to_numeric(df["person_detected"], errors="coerce")
        .fillna(0) == 1
    )

    for joint in joints:
        conf = pd.to_numeric(
            df[f"{joint}_confidence"], errors="coerce"
        )
        x = pd.to_numeric(df[f"{joint}_x"], errors="coerce").astype(float)
        y = pd.to_numeric(df[f"{joint}_y"], errors="coerce").astype(float)

        valid = (
            (conf >= CONFIDENCE_THRESHOLD)
            & detected
            & x.notna()
            & y.notna()
        )

        x.loc[~valid] = np.nan
        y.loc[~valid] = np.nan

        x = x.interpolate(
            method="linear", limit=MAX_INTERPOLATION_GAP,
            limit_direction="both"
        )
        y = y.interpolate(
            method="linear", limit=MAX_INTERPOLATION_GAP,
            limit_direction="both"
        )

        x = x.rolling(
            TEMPORAL_SMOOTHING_WINDOW, center=True, min_periods=1
        ).median()
        y = y.rolling(
            TEMPORAL_SMOOTHING_WINDOW, center=True, min_periods=1
        ).median()

        prepared[joint] = (x.to_numpy(), y.to_numpy())

    shoulder_x = (
        prepared["left_shoulder"][0] + prepared["right_shoulder"][0]
    ) / 2
    shoulder_y = (
        prepared["left_shoulder"][1] + prepared["right_shoulder"][1]
    ) / 2
    hip_x = (
        prepared["left_hip"][0] + prepared["right_hip"][0]
    ) / 2
    hip_y = (
        prepared["left_hip"][1] + prepared["right_hip"][1]
    ) / 2

    torso = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y)
    torso[~np.isfinite(torso) | (torso < 1e-6)] = np.nan

    features = pd.DataFrame({"frame": df["frame"].to_numpy()})

    for joint, (x, y) in prepared.items():
        norm_x = (x - hip_x) / torso
        norm_y = (y - hip_y) / torso

        vx = np.gradient(norm_x) * FPS
        vy = np.gradient(norm_y) * FPS
        ax = np.gradient(vx) * FPS
        ay = np.gradient(vy) * FPS

        features[f"{joint}_norm_x"] = norm_x
        features[f"{joint}_norm_y"] = norm_y
        features[f"{joint}_x_velocity"] = vx
        features[f"{joint}_y_velocity"] = vy
        features[f"{joint}_x_acceleration"] = ax
        features[f"{joint}_y_acceleration"] = ay
        features[f"{joint}_speed"] = np.hypot(vx, vy)
        features[f"{joint}_acceleration_magnitude"] = np.hypot(ax, ay)

    return features


def extract_features(keypoints_csv, width=1920, height=1080):
    raw = pd.read_csv(keypoints_csv)

    required = {"frame", "person_detected"}
    for joint in JOINTS:
        required.update({
            f"{joint}_x", f"{joint}_y", f"{joint}_confidence"
        })

    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Keypoints CSV is missing columns: {missing}")

    raw = raw.sort_values("frame").reset_index(drop=True)

    confidence = build_confidence_coordinates(raw, width, height)
    lower = build_lower_body_features(confidence)
    temporal = prepare_temporal_coordinates(raw)

    combined = lower.merge(
        temporal, on="frame", how="inner", suffixes=("", "_temporal")
    )

    temporal_names = [
        "left_hip_x_velocity", "left_hip_y_velocity",
        "right_hip_x_velocity", "right_hip_y_velocity",
        "left_knee_x_velocity", "left_knee_y_velocity",
        "right_knee_x_velocity", "right_knee_y_velocity",
        "left_ankle_x_velocity", "left_ankle_y_velocity",
        "right_ankle_x_velocity", "right_ankle_y_velocity",
        "left_knee_acceleration_magnitude",
        "right_knee_acceleration_magnitude",
        "left_ankle_acceleration_magnitude",
        "right_ankle_acceleration_magnitude"
    ]

    for name in temporal_names:
        combined[f"temporal_{name}"] = combined[name]

    values = combined[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(
        method="linear", limit_direction="both"
    ).fillna(0.0)

    return combined[["frame"]].join(values)


def make_sequences(feature_df):
    values = feature_df[FEATURES].to_numpy(dtype=np.float32)

    sequences = []
    frame_ranges = []

    for start in range(0, len(values) - SEQUENCE_LENGTH + 1, STRIDE):
        end = start + SEQUENCE_LENGTH
        sequences.append(values[start:end])
        frame_ranges.append([
            int(feature_df.iloc[start]["frame"]),
            int(feature_df.iloc[end - 1]["frame"])
        ])

    if not sequences:
        raise ValueError(
            f"Video has {len(values)} usable frames; "
            f"at least {SEQUENCE_LENGTH} are required."
        )

    return np.stack(sequences), frame_ranges


def main():
    parser = argparse.ArgumentParser(
        description="Extract GaitXplain model features from YOLO keypoints."
    )
    parser.add_argument("--input", required=True, help="YOLO keypoints CSV")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    feature_df = extract_features(
        args.input, width=args.width, height=args.height
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output, index=False)

    print(f"Saved features: {output}")
    print(f"Frames: {len(feature_df)}")
    print(f"Feature count: {len(FEATURES)}")
    print(f"Missing values: {int(feature_df[FEATURES].isna().sum().sum())}")


if __name__ == "__main__":
    main()

