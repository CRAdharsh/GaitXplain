#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/Final Year Project/GAVD-main"
)

DATASET_DIR = BASE_DIR / "full_dataset" / "v2_2sec"

METADATA_FILE = DATASET_DIR / "metadata.csv"
SEQUENCE_DIR = DATASET_DIR / "sequences"

REPORT_FILE = DATASET_DIR / "audit_report.json"
BAD_FILES_FILE = DATASET_DIR / "bad_npz_files.csv"
BAD_VIDEOS_FILE = DATASET_DIR / "videos_with_problems.csv"


# Expected representation
EXPECTED_TOKENS = 32
EXPECTED_HEIGHT = 256
EXPECTED_WIDTH = 128
EXPECTED_KEYPOINTS = 17
EXPECTED_KEYPOINT_FEATURES = 3

EXPECTED_CLASSES = [
    "abnormal",
    "antalgic",
    "cerebral palsy",
    "exercise",
    "inebriated",
    "myopathic",
    "normal",
    "parkinsons",
    "pregnant",
    "prosthetic",
    "stroke",
    "style",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def is_finite_array(arr):
    if not np.issubdtype(arr.dtype, np.number):
        return True

    return bool(np.isfinite(arr).all())


def percentage(a, b):
    if b == 0:
        return 0.0
    return 100.0 * a / b


# ============================================================
# CHECK DATASET STRUCTURE
# ============================================================

print("=" * 80)
print("GAVD V2 DATASET AUDIT")
print("=" * 80)

print()
print("Dataset directory:")
print(DATASET_DIR)

print()
print("Metadata:")
print(METADATA_FILE)

print()
print("Sequence directory:")
print(SEQUENCE_DIR)

if not DATASET_DIR.exists():
    print("\nERROR: Dataset directory does not exist.")
    sys.exit(1)

if not METADATA_FILE.exists():
    print("\nERROR: metadata.csv does not exist.")
    sys.exit(1)

if not SEQUENCE_DIR.exists():
    print("\nERROR: sequences directory does not exist.")
    sys.exit(1)


# ============================================================
# LOAD METADATA
# ============================================================

print()
print("-" * 80)
print("1. METADATA AUDIT")
print("-" * 80)

try:
    metadata = pd.read_csv(METADATA_FILE)
except Exception as e:
    print(f"ERROR reading metadata.csv: {e}")
    sys.exit(1)

print(f"Metadata rows: {len(metadata):,}")
print(f"Metadata columns: {len(metadata.columns)}")

print("\nColumns:")
for col in metadata.columns:
    print(f"  - {col}")


# ============================================================
# REQUIRED METADATA COLUMNS
# ============================================================

required_columns = [
    "video_id",
    "sequence_id",
    "gait_class",
]

missing_columns = [
    col for col in required_columns
    if col not in metadata.columns
]

if missing_columns:
    print("\nWARNING: Missing required columns:")
    for col in missing_columns:
        print(f"  - {col}")
else:
    print("\nRequired metadata columns: OK")


# ============================================================
# BASIC METADATA COUNTS
# ============================================================

if "video_id" in metadata.columns:
    metadata_video_ids = set(
        metadata["video_id"]
        .dropna()
        .astype(str)
    )
else:
    metadata_video_ids = set()

if "sequence_id" in metadata.columns:
    metadata_sequence_ids = set(
        metadata["sequence_id"]
        .dropna()
        .astype(str)
    )
else:
    metadata_sequence_ids = set()

print()
print(f"Unique videos in metadata:    {len(metadata_video_ids):,}")
print(f"Unique sequences in metadata: {len(metadata_sequence_ids):,}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("-" * 80)
print("2. CLASS DISTRIBUTION")
print("-" * 80)

class_counts = {}

if "gait_class" in metadata.columns:

    class_counts = (
        metadata["gait_class"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    total_rows = len(metadata)

    print()

    for cls, count in sorted(class_counts.items()):
        print(
            f"{cls:<20} "
            f"{count:>6,} "
            f"({percentage(count, total_rows):6.2f}%)"
        )

    print()
    print("Expected GAVD classes missing from metadata:")

    missing_classes = [
        cls for cls in EXPECTED_CLASSES
        if cls not in class_counts
    ]

    if missing_classes:
        for cls in missing_classes:
            print(f"  WARNING: {cls}")
    else:
        print("  None")


# ============================================================
# NPZ FILE DISCOVERY
# ============================================================

print()
print("-" * 80)
print("3. NPZ FILE AUDIT")
print("-" * 80)

npz_files = sorted(SEQUENCE_DIR.glob("*.npz"))

print(f"NPZ files found: {len(npz_files):,}")

if len(npz_files) == 0:
    print("\nERROR: No NPZ files found.")
    sys.exit(1)


# ============================================================
# EXPECTED NPZ ARRAYS
# ============================================================

expected_arrays = [
    "edges",
    "keypoints",
    "frame_numbers",
    "timestamps",
    "relative_timestamps",
    "delta_t",
    "source_fps",
]


# ============================================================
# NPZ AUDIT
# ============================================================

bad_files = []

total_edges = 0
total_keypoints = 0

segmentation_invalid_windows = 0
pose_invalid_windows = 0

segmentation_valid_windows = 0
pose_valid_windows = 0

both_valid_windows = 0

nan_windows = 0
inf_windows = 0

shape_errors = 0
missing_array_errors = 0
load_errors = 0

fps_counter = Counter()

window_lengths = Counter()

videos_from_npz = set()
sequences_from_npz = set()

video_window_counts = Counter()
sequence_window_counts = Counter()

class_window_counts = Counter()

video_class_map = {}

temporal_errors = []

frame_number_errors = []

source_fps_errors = []

zero_edge_windows = 0
zero_keypoint_windows = 0

print("\nScanning NPZ files...")

for index, npz_path in enumerate(npz_files, start=1):

    file_problems = []

    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:

        load_errors += 1

        bad_files.append({
            "file": str(npz_path),
            "problems": [f"LOAD_ERROR: {e}"]
        })

        continue

    # --------------------------------------------------------
    # Check required arrays
    # --------------------------------------------------------

    missing_arrays = [
        name for name in expected_arrays
        if name not in data.files
    ]

    if missing_arrays:

        missing_array_errors += 1

        file_problems.append(
            "MISSING_ARRAYS: " + ",".join(missing_arrays)
        )

    # --------------------------------------------------------
    # Required arrays available?
    # --------------------------------------------------------

    if "edges" not in data.files:
        bad_files.append({
            "file": str(npz_path),
            "problems": file_problems
        })
        continue

    edges = data["edges"]

    if "keypoints" in data.files:
        keypoints = data["keypoints"]
    else:
        keypoints = None

    if "frame_numbers" in data.files:
        frame_numbers = data["frame_numbers"]
    else:
        frame_numbers = None

    if "timestamps" in data.files:
        timestamps = data["timestamps"]
    else:
        timestamps = None

    if "relative_timestamps" in data.files:
        relative_timestamps = data["relative_timestamps"]
    else:
        relative_timestamps = None

    if "delta_t" in data.files:
        delta_t = data["delta_t"]
    else:
        delta_t = None

    if "source_fps" in data.files:
        source_fps = data["source_fps"]
    else:
        source_fps = None

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if edges.shape != (
        EXPECTED_TOKENS,
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
    ):

        shape_errors += 1

        file_problems.append(
            f"BAD_EDGE_SHAPE: {edges.shape}"
        )

    if keypoints is not None:

        if keypoints.shape != (
            EXPECTED_TOKENS,
            EXPECTED_KEYPOINTS,
            EXPECTED_KEYPOINT_FEATURES,
        ):

            shape_errors += 1

            file_problems.append(
                f"BAD_KEYPOINT_SHAPE: {keypoints.shape}"
            )

    # --------------------------------------------------------
    # Temporal length
    # --------------------------------------------------------

    if frame_numbers is not None:

        if len(frame_numbers) != EXPECTED_TOKENS:
            file_problems.append(
                f"BAD_FRAME_COUNT: {len(frame_numbers)}"
            )

    if timestamps is not None:

        if len(timestamps) != EXPECTED_TOKENS:
            file_problems.append(
                f"BAD_TIMESTAMP_COUNT: {len(timestamps)}"
            )

    if relative_timestamps is not None:

        if len(relative_timestamps) != EXPECTED_TOKENS:
            file_problems.append(
                f"BAD_RELATIVE_TIMESTAMP_COUNT: {len(relative_timestamps)}"
            )

    if delta_t is not None:

        if len(delta_t) != EXPECTED_TOKENS:
            file_problems.append(
                f"BAD_DELTA_T_COUNT: {len(delta_t)}"
            )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    arrays_to_check = [
        ("edges", edges),
        ("keypoints", keypoints),
        ("frame_numbers", frame_numbers),
        ("timestamps", timestamps),
        ("relative_timestamps", relative_timestamps),
        ("delta_t", delta_t),
    ]

    for array_name, arr in arrays_to_check:

        if arr is None:
            continue

        if np.issubdtype(arr.dtype, np.number):

            finite = np.isfinite(arr)

            if not finite.all():

                if np.isnan(arr).any():
                    nan_windows += 1
                    file_problems.append(
                        f"NAN_VALUES_IN_{array_name}"
                    )

                if np.isinf(arr).any():
                    inf_windows += 1
                    file_problems.append(
                        f"INF_VALUES_IN_{array_name}"
                    )

    # --------------------------------------------------------
    # Edge validity
    # --------------------------------------------------------

    if edges.ndim == 3 and len(edges) == EXPECTED_TOKENS:

        edge_nonzero = np.any(edges != 0, axis=(1, 2))

        valid_count = int(edge_nonzero.sum())

        invalid_count = EXPECTED_TOKENS - valid_count

        total_edges += valid_count

        if invalid_count > 0:

            segmentation_invalid_windows += 1

            if valid_count == 0:
                zero_edge_windows += 1

        else:
            segmentation_valid_windows += 1

    # --------------------------------------------------------
    # Keypoint validity
    # --------------------------------------------------------

    if keypoints is not None:

        if keypoints.ndim == 3 and len(keypoints) == EXPECTED_TOKENS:

            # Pose validity is determined by confidence channel.
            # confidence > 0 means at least one detected keypoint.
            confidence = keypoints[:, :, 2]

            valid_pose_frames = np.any(
                confidence > 0,
                axis=1
            )

            valid_count = int(valid_pose_frames.sum())

            invalid_count = EXPECTED_TOKENS - valid_count

            total_keypoints += valid_count

            if invalid_count > 0:

                pose_invalid_windows += 1

                if valid_count == 0:
                    zero_keypoint_windows += 1

            else:
                pose_valid_windows += 1

    # --------------------------------------------------------
    # Combined validity
    # --------------------------------------------------------

    edge_ok = (
        edges.ndim == 3
        and len(edges) == EXPECTED_TOKENS
        and np.all(np.any(edges != 0, axis=(1, 2)))
    )

    pose_ok = False

    if keypoints is not None:

        if keypoints.ndim == 3 and len(keypoints) == EXPECTED_TOKENS:

            pose_ok = np.all(
                np.any(keypoints[:, :, 2] > 0, axis=1)
            )

    if edge_ok and pose_ok:
        both_valid_windows += 1

    # --------------------------------------------------------
    # Temporal consistency
    # --------------------------------------------------------

    if timestamps is not None:

        if len(timestamps) == EXPECTED_TOKENS:

            if not np.all(np.diff(timestamps) >= 0):

                temporal_errors.append(
                    str(npz_path)
                )

                file_problems.append(
                    "TIMESTAMPS_NOT_MONOTONIC"
                )

    if relative_timestamps is not None:

        if len(relative_timestamps) == EXPECTED_TOKENS:

            if not np.all(np.diff(relative_timestamps) >= 0):

                temporal_errors.append(
                    str(npz_path)
                )

                file_problems.append(
                    "RELATIVE_TIMESTAMPS_NOT_MONOTONIC"
                )

    if delta_t is not None:

        if len(delta_t) == EXPECTED_TOKENS:

            if np.any(delta_t < 0):

                temporal_errors.append(
                    str(npz_path)
                )

                file_problems.append(
                    "NEGATIVE_DELTA_T"
                )

    # --------------------------------------------------------
    # Frame number consistency
    # --------------------------------------------------------

    if frame_numbers is not None:

        if len(frame_numbers) == EXPECTED_TOKENS:

            if not np.all(np.diff(frame_numbers) > 0):

                frame_number_errors.append(
                    str(npz_path)
                )

                file_problems.append(
                    "FRAME_NUMBERS_NOT_STRICTLY_INCREASING"
                )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if source_fps is not None:

        try:

            fps_value = float(np.asarray(source_fps).reshape(-1)[0])

            if fps_value > 0 and math.isfinite(fps_value):

                fps_counter[round(fps_value, 3)] += 1

            else:

                source_fps_errors.append(
                    str(npz_path)
                )

                file_problems.append(
                    "INVALID_SOURCE_FPS"
                )

        except Exception:

            source_fps_errors.append(
                str(npz_path)
            )

            file_problems.append(
                "INVALID_SOURCE_FPS"
            )

    # --------------------------------------------------------
    # Extract video/sequence/class from metadata columns
    # --------------------------------------------------------

    if "video_id" in data.files:

        try:
            video_id = str(np.asarray(data["video_id"]).item())
        except Exception:
            video_id = None

    else:
        video_id = None

    if "sequence_id" in data.files:

        try:
            sequence_id = str(np.asarray(data["sequence_id"]).item())
        except Exception:
            sequence_id = None

    else:
        sequence_id = None

    if "gait_class" in data.files:

        try:
            gait_class = str(np.asarray(data["gait_class"]).item())
        except Exception:
            gait_class = None

    else:
        gait_class = None

    # If metadata isn't stored inside NPZ,
    # infer it from metadata.csv using filename.

    if video_id is None and "video_id" in metadata.columns:

        matches = metadata[
            metadata["npz_file"].astype(str) == npz_path.name
        ] if "npz_file" in metadata.columns else pd.DataFrame()

        if len(matches) > 0:

            video_id = str(matches.iloc[0]["video_id"])

            if "sequence_id" in matches.columns:
                sequence_id = str(matches.iloc[0]["sequence_id"])

            if "gait_class" in matches.columns:
                gait_class = str(matches.iloc[0]["gait_class"])

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if video_id is not None:

        videos_from_npz.add(video_id)
        video_window_counts[video_id] += 1

    if sequence_id is not None:

        sequences_from_npz.add(sequence_id)
        sequence_window_counts[sequence_id] += 1

    if gait_class is not None:

        class_window_counts[gait_class] += 1

    # --------------------------------------------------------
    # Save problems
    # --------------------------------------------------------

    if file_problems:

        bad_files.append({
            "file": str(npz_path),
            "problems": file_problems
        })

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 500 == 0 or index == len(npz_files):

        print(
            f"  Processed {index:,}/{len(npz_files):,}"
        )


# ============================================================
# METADATA vs NPZ COMPARISON
# ============================================================

print()
print("-" * 80)
print("4. METADATA ↔ NPZ CONSISTENCY")
print("-" * 80)

metadata_npz_names = set()

if "npz_file" in metadata.columns:

    metadata_npz_names = set(
        metadata["npz_file"]
        .dropna()
        .astype(str)
    )

else:

    # If metadata doesn't contain npz_file,
    # compare number of rows instead.
    metadata_npz_names = set()


npz_names = set(
    p.name for p in npz_files
)

if metadata_npz_names:

    missing_npz_from_disk = metadata_npz_names - npz_names
    unlisted_npz = npz_names - metadata_npz_names

    print(
        f"Metadata NPZ references: {len(metadata_npz_names):,}"
    )

    print(
        f"NPZ files on disk:       {len(npz_names):,}"
    )

    print(
        f"Referenced but missing:   {len(missing_npz_from_disk):,}"
    )

    print(
        f"On disk but not metadata: {len(unlisted_npz):,}"
    )

else:

    print(
        "metadata.csv has no 'npz_file' column."
    )

    print(
        "Using row count vs NPZ count instead."
    )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    print(
        f"NPZ files:     {len(npz_files):,}"
    )


# ============================================================
# VIDEO COVERAGE
# ============================================================

print()
print("-" * 80)
print("5. VIDEO COVERAGE")
print("-" * 80)

if "video_id" in metadata.columns:

    all_metadata_videos = set(
        metadata["video_id"]
        .dropna()
        .astype(str)
    )

    videos_without_npz = (
        all_metadata_videos
        - videos_from_npz
    )

    videos_with_npz = (
        all_metadata_videos
        & videos_from_npz
    )

    print(
        f"Videos in metadata:       {len(all_metadata_videos):,}"
    )

    print(
        f"Videos with NPZ data:     {len(videos_with_npz):,}"
    )

    print(
        f"Videos without NPZ data:  {len(videos_without_npz):,}"
    )

    if videos_without_npz:

        print("\nVideos WITHOUT generated data:")

        for video in sorted(videos_without_npz):
            print(f"  {video}")


# ============================================================
# SEQUENCE COVERAGE
# ============================================================

print()
print("-" * 80)
print("6. SEQUENCE COVERAGE")
print("-" * 80)

if "sequence_id" in metadata.columns:

    all_metadata_sequences = set(
        metadata["sequence_id"]
        .dropna()
        .astype(str)
    )

    sequences_without_npz = (
        all_metadata_sequences
        - sequences_from_npz
    )

    print(
        f"Sequences in metadata:      {len(all_metadata_sequences):,}"
    )

    print(
        f"Sequences with NPZ windows:  {len(sequences_from_npz):,}"
    )

    print(
        f"Sequences without windows:   {len(sequences_without_npz):,}"
    )


# ============================================================
# WINDOW VALIDITY
# ============================================================

print()
print("-" * 80)
print("7. WINDOW VALIDITY")
print("-" * 80)

print(
    f"Total windows:                       {len(npz_files):,}"
)

print(
    f"Windows with all edge frames valid:  {segmentation_valid_windows:,}"
)

print(
    f"Windows with all pose frames valid:  {pose_valid_windows:,}"
)

print(
    f"Windows with BOTH fully valid:       {both_valid_windows:,}"
)

print(
    f"Windows containing edge failures:     {segmentation_invalid_windows:,}"
)

print(
    f"Windows containing pose failures:     {pose_invalid_windows:,}"
)

print(
    f"Windows with ALL edges zero:          {zero_edge_windows:,}"
)

print(
    f"Windows with ALL pose invalid:        {zero_keypoint_windows:,}"
)

print()
print(
    f"Fully valid percentage: "
    f"{percentage(both_valid_windows, len(npz_files)):.2f}%"
)


# ============================================================
# TEMPORAL AUDIT
# ============================================================

print()
print("-" * 80)
print("8. TEMPORAL CONSISTENCY")
print("-" * 80)

print(
    f"Timestamp errors:       {len(set(temporal_errors)):,}"
)

print(
    f"Frame-number errors:    {len(set(frame_number_errors)):,}"
)

print(
    f"FPS errors:             {len(set(source_fps_errors)):,}"
)


# ============================================================
# CLASS WINDOW DISTRIBUTION
# ============================================================

print()
print("-" * 80)
print("9. WINDOW CLASS DISTRIBUTION")
print("-" * 80)

if class_window_counts:

    total_class_windows = sum(
        class_window_counts.values()
    )

    for cls, count in sorted(
        class_window_counts.items(),
        key=lambda x: x[0]
    ):

        print(
            f"{cls:<20} "
            f"{count:>7,} "
            f"({percentage(count, total_class_windows):6.2f}%)"
        )


# ============================================================
# VIDEO WINDOW DISTRIBUTION
# ============================================================

print()
print("-" * 80)
print("10. WINDOWS PER VIDEO")
print("-" * 80)

if video_window_counts:

    values = list(video_window_counts.values())

    print(f"Videos with windows: {len(values):,}")
    print(f"Minimum windows/video: {min(values):,}")
    print(f"Maximum windows/video: {max(values):,}")
    print(f"Mean windows/video:    {np.mean(values):.2f}")
    print(f"Median windows/video:  {np.median(values):.2f}")


# ============================================================
# FILE PROBLEM SUMMARY
# ============================================================

print()
print("-" * 80)
print("11. FILE PROBLEMS")
print("-" * 80)

print(f"Files with problems: {len(bad_files):,}")
print(f"Load errors:         {load_errors:,}")
print(f"Shape errors:        {shape_errors:,}")
print(f"Missing arrays:      {missing_array_errors:,}")
print(f"NaN occurrences:     {nan_windows:,}")
print(f"Inf occurrences:     {inf_windows:,}")


# ============================================================
# SAVE BAD FILE REPORT
# ============================================================

if bad_files:

    rows = []

    for item in bad_files:

        rows.append({
            "file": item["file"],
            "problems": " | ".join(item["problems"])
        })

    bad_df = pd.DataFrame(rows)

    bad_df.to_csv(
        BAD_FILES_FILE,
        index=False
    )

    print()
    print(
        f"Bad-file report saved to:\n{BAD_FILES_FILE}"
    )


# ============================================================
# SAVE VIDEO PROBLEM REPORT
# ============================================================

video_problem_rows = []

if "video_id" in metadata.columns:

    all_metadata_videos = set(
        metadata["video_id"]
        .dropna()
        .astype(str)
    )

    for video_id in sorted(all_metadata_videos):

        window_count = video_window_counts.get(
            video_id,
            0
        )

        if window_count == 0:

            rows = metadata[
                metadata["video_id"].astype(str)
                == video_id
            ]

            gait_class = (
                rows["gait_class"].iloc[0]
                if "gait_class" in rows.columns
                and len(rows) > 0
                else ""
            )

            video_problem_rows.append({
                "video_id": video_id,
                "gait_class": gait_class,
                "windows": 0,
                "problem": "NO_GENERATED_WINDOWS"
            })


if video_problem_rows:

    pd.DataFrame(
        video_problem_rows
    ).to_csv(
        BAD_VIDEOS_FILE,
        index=False
    )

    print(
        f"Video problem report saved to:\n{BAD_VIDEOS_FILE}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

report = {

    "dataset_directory": str(DATASET_DIR),

    "metadata": {
        "rows": int(len(metadata)),
        "columns": list(metadata.columns),
        "unique_videos": int(
            len(metadata_video_ids)
        ),
        "unique_sequences": int(
            len(metadata_sequence_ids)
        ),
    },

    "npz": {
        "files": int(len(npz_files)),
        "load_errors": int(load_errors),
        "shape_errors": int(shape_errors),
        "missing_array_errors": int(
            missing_array_errors
        ),
    },

    "classes": {
        "metadata_distribution": {
            str(k): int(v)
            for k, v in class_counts.items()
        },
        "window_distribution": {
            str(k): int(v)
            for k, v in class_window_counts.items()
        },
        "missing_expected_classes": [
            cls for cls in EXPECTED_CLASSES
            if cls not in class_counts
        ],
    },

    "coverage": {

        "videos_in_metadata": int(
            len(metadata_video_ids)
        ),

        "videos_with_npz": int(
            len(videos_from_npz)
        ),

        "videos_without_npz": int(
            len(
                metadata_video_ids
                - videos_from_npz
            )
        ),

        "sequences_in_metadata": int(
            len(metadata_sequence_ids)
        ),

        "sequences_with_npz": int(
            len(sequences_from_npz)
        ),
    },

    "windows": {

        "total": int(len(npz_files)),

        "fully_valid_edges": int(
            segmentation_valid_windows
        ),

        "fully_valid_pose": int(
            pose_valid_windows
        ),

        "fully_valid_both": int(
            both_valid_windows
        ),

        "edge_failure_windows": int(
            segmentation_invalid_windows
        ),

        "pose_failure_windows": int(
            pose_invalid_windows
        ),

        "all_edges_zero": int(
            zero_edge_windows
        ),

        "all_pose_invalid": int(
            zero_keypoint_windows
        ),
    },

    "temporal": {

        "timestamp_errors": int(
            len(set(temporal_errors))
        ),

        "frame_number_errors": int(
            len(set(frame_number_errors))
        ),

        "fps_errors": int(
            len(set(source_fps_errors))
        ),
    },

    "problems": {

        "files_with_problems": int(
            len(bad_files)
        ),

        "nan_occurrences": int(
            nan_windows
        ),

        "inf_occurrences": int(
            inf_windows
        ),
    },

    "expected_shapes": {

        "edges": [
            EXPECTED_TOKENS,
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH
        ],

        "keypoints": [
            EXPECTED_TOKENS,
            EXPECTED_KEYPOINTS,
            EXPECTED_KEYPOINT_FEATURES
        ],
    },
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print()
print(f"NPZ windows:                 {len(npz_files):,}")
print(f"Videos with data:            {len(videos_from_npz):,}")
print(f"Sequences with data:         {len(sequences_from_npz):,}")
print(
    f"Fully valid windows:         "
    f"{both_valid_windows:,} "
    f"({percentage(both_valid_windows, len(npz_files)):.2f}%)"
)
print(
    f"Windows with edge failures:  "
    f"{segmentation_invalid_windows:,}"
)
print(
    f"Windows with pose failures:  "
    f"{pose_invalid_windows:,}"
)
print(
    f"Temporal errors:             "
    f"{len(set(temporal_errors)):,}"
)
print(
    f"Frame-number errors:         "
    f"{len(set(frame_number_errors)):,}"
)
print(
    f"Files with problems:         "
    f"{len(bad_files):,}"
)

print()
print("Reports:")
print(f"  {REPORT_FILE}")

if bad_files:
    print(f"  {BAD_FILES_FILE}")

if video_problem_rows:
    print(f"  {BAD_VIDEOS_FILE}")

print()
print("=" * 80)