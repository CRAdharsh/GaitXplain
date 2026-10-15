#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

DATASET_DIR = Path(
    "/Final Year Project/GAVD-main/full_dataset/v2_2sec"
)

METADATA_FILE = DATASET_DIR / "metadata.csv"

CLASS_MAPPING_FILE = DATASET_DIR / "class_mapping.json"
DATASET_CONFIG_FILE = DATASET_DIR / "dataset_config.json"


# ============================================================
# OFFICIAL GAVD CLASS ORDER
# ============================================================

CLASSES = [
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
# START
# ============================================================

print("=" * 80)
print("CREATING DATASET INFORMATION FILES")
print("=" * 80)


# ============================================================
# CHECK FILES
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory does not exist:\n{DATASET_DIR}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"metadata.csv not found:\n{METADATA_FILE}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

print("\nReading metadata.csv...")

metadata = pd.read_csv(METADATA_FILE)

print(f"Metadata rows: {len(metadata):,}")

print("\nMetadata columns:")

for column in metadata.columns:
    print(f"  - {column}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "filename",
    "video_id",
    "sequence_id",
    "gait_pat",
    "cam_view",
    "window_start",
    "window_end",
    "source_fps",
    "tokens",
    "segmentation_valid_tokens",
    "pose_valid_tokens",
]

missing_columns = [
    column
    for column in required_columns
    if column not in metadata.columns
]

if missing_columns:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )

print("\nRequired metadata columns: OK")


# ============================================================
# CLASS CHECK
# ============================================================

metadata_classes = sorted(
    metadata["gait_pat"]
    .dropna()
    .astype(str)
    .unique()
)

print("\nClasses found in metadata:")

for cls in metadata_classes:
    count = int(
        (metadata["gait_pat"].astype(str) == cls).sum()
    )

    print(
        f"  {cls:<20} {count:>6,}"
    )


missing_classes = [
    cls
    for cls in CLASSES
    if cls not in metadata_classes
]

unexpected_classes = [
    cls
    for cls in metadata_classes
    if cls not in CLASSES
]

if missing_classes:

    print("\nWARNING: Expected classes missing:")

    for cls in missing_classes:
        print(f"  - {cls}")

else:

    print("\nAll 12 expected GAVD classes are present.")


if unexpected_classes:

    print("\nWARNING: Unexpected classes found:")

    for cls in unexpected_classes:
        print(f"  - {cls}")


# ============================================================
# CLASS MAPPING
# ============================================================

class_mapping = {
    cls: index
    for index, cls in enumerate(CLASSES)
}

class_mapping_data = {

    "description":
        "GAVD gait-pattern to integer-label mapping",

    "label_column":
        "gait_pat",

    "num_classes":
        len(CLASSES),

    "classes":
        class_mapping
}


with open(
    CLASS_MAPPING_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_mapping_data,
        file,
        indent=4
    )


print(
    f"\nCreated:\n{CLASS_MAPPING_FILE}"
)


# ============================================================
# DATASET STATISTICS
# ============================================================

num_windows = len(metadata)

num_videos = metadata["video_id"].nunique()

num_sequences = metadata["sequence_id"].nunique()

num_camera_views = metadata["cam_view"].nunique()

source_fps_values = (
    metadata["source_fps"]
    .dropna()
    .unique()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    metadata["gait_pat"]
    .dropna()
    .astype(str)
    .value_counts()
    .to_dict()
)


# ============================================================
# VALIDITY STATISTICS
# ============================================================

fully_valid_segmentation = int(
    (
        metadata["segmentation_valid_tokens"]
        == metadata["tokens"]
    ).sum()
)

fully_valid_pose = int(
    (
        metadata["pose_valid_tokens"]
        == metadata["tokens"]
    ).sum()
)

fully_valid_both = int(
    (
        (metadata["segmentation_valid_tokens"] == metadata["tokens"])
        &
        (metadata["pose_valid_tokens"] == metadata["tokens"])
    ).sum()
)


# ============================================================
# DATASET CONFIG
# ============================================================

dataset_config = {

    "dataset_name":
        "GAVD Preprocessed V2",

    "source_dataset":
        "Gait Abnormality Video Dataset (GAVD)",

    "description":
        (
            "Preprocessed GAVD dataset containing "
            "physical-time gait windows represented by "
            "edge maps, human keypoints, and original "
            "temporal information."
        ),

    "status":
        "UNSPLIT",

    "label_column":
        "gait_pat",

    "preprocessing": {

        "window_seconds":
            2.0,

        "overlap":
            0.5,

        "overlap_percentage":
            50,

        "tokens_per_window":
            32,

        "interpolation":
            False,

        "temporal_representation":
            "physical-time windows with original source timestamps",

        "edge_representation": {

            "shape":
                [32, 256, 128],

            "segmentation_model":
                "YOLO11n-seg",

            "person_selection":
                "largest detected person",

            "polygon_mask":
                True,

            "morphological_closing":
                {
                    "kernel_size":
                        [5, 5],

                    "kernel_type":
                        "ellipse"
                },

            "largest_connected_component":
                True,

            "minimum_component_area":
                500,

            "roi_padding_pixels":
                10,

            "resize":
                [256, 128],

            "resize_method":
                "aspect-preserving resize with padding",

            "canny":
                {
                    "lower_threshold":
                        50,

                    "upper_threshold":
                        150
                },

            "dtype":
                "uint8"
        },

        "pose_representation": {

            "shape":
                [32, 17, 3],

            "pose_model":
                "YOLO11n-pose",

            "keypoints":
                17,

            "features_per_keypoint":
                [
                    "x",
                    "y",
                    "confidence"
                ],

            "dtype":
                "float32"
        },

        "temporal_arrays": {

            "frame_numbers":
                {
                    "shape": [32],
                    "description":
                        "Original source frame numbers"
                },

            "timestamps":
                {
                    "shape": [32],
                    "description":
                        "Original source timestamps"
                },

            "relative_timestamps":
                {
                    "shape": [32],
                    "description":
                        "Timestamps relative to the beginning of the window"
                },

            "delta_t":
                {
                    "shape": [32],
                    "description":
                        "Temporal difference information between frames"
                },

            "source_fps":
                {
                    "description":
                        "Original source video FPS"
                }
        }
    },

    "classes": {

        "num_classes":
            len(CLASSES),

        "label_column":
            "gait_pat",

        "class_mapping_file":
            "class_mapping.json",

        "class_names":
            CLASSES
    },

    "dataset_statistics": {

        "metadata_rows":
            int(num_windows),

        "unique_videos":
            int(num_videos),

        "unique_sequences":
            int(num_sequences),

        "unique_camera_views":
            int(num_camera_views),

        "unique_source_fps_values":
            sorted(
                float(x)
                for x in source_fps_values
            ),

        "class_distribution":
            {
                str(k): int(v)
                for k, v in class_counts.items()
            },

        "fully_valid_segmentation_windows":
            fully_valid_segmentation,

        "fully_valid_pose_windows":
            fully_valid_pose,

        "fully_valid_both_windows":
            fully_valid_both
    },

    "metadata_columns": list(
        metadata.columns
    ),

    "data_format": {

        "metadata":
            "metadata.csv",

        "sample_format":
            "NPZ",

        "sample_directory":
            "sequences/",

        "npz_arrays": [

            "edges",

            "keypoints",

            "frame_numbers",

            "timestamps",

            "relative_timestamps",

            "delta_t",

            "source_fps"
        ]
    },

    "important_notes": [

        "This dataset has not been split into training, validation, and test sets.",

        "Train/validation/test splitting should be performed at the video_id level.",

        "Windows from the same video should not be distributed across different dataset splits.",

        "Original source FPS and temporal information are preserved.",

        "Frames are not interpolated to a common FPS.",

        "The 32 temporal tokens represent the model input representation.",

        "gait_pat is the class-label column in metadata.csv.",

        "The dataset contains the 12 GAVD gait-pattern classes represented in the metadata."
    ]
}


# ============================================================
# WRITE CONFIG
# ============================================================

with open(
    DATASET_CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        dataset_config,
        file,
        indent=4
    )


print(
    f"Created:\n{DATASET_CONFIG_FILE}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("DATASET INFORMATION CREATED")
print("=" * 80)

print()
print(f"Dataset directory:")
print(f"  {DATASET_DIR}")

print()
print("Created:")

print(f"  ✓ {CLASS_MAPPING_FILE.name}")
print(f"  ✓ {DATASET_CONFIG_FILE.name}")

print()
print("Dataset statistics:")

print(f"  Windows:              {num_windows:,}")
print(f"  Videos:               {num_videos:,}")
print(f"  Sequences:            {num_sequences:,}")
print(f"  Classes:              {len(CLASSES)}")

print()
print("Validity:")

print(
    f"  Fully valid segmentation: "
    f"{fully_valid_segmentation:,}"
)

print(
    f"  Fully valid pose:         "
    f"{fully_valid_pose:,}"
)

print(
    f"  Fully valid both:         "
    f"{fully_valid_both:,}"
)

print()
print("Existing dataset files were NOT modified.")

print("=" * 80)