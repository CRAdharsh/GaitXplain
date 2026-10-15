import pandas as pd
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

BASE = "/Final Year Project"

ANNOTATION_FILE = os.path.join(
    BASE,
    "GAVD-main",
    "analysis",
    "GAVD_combined_annotations.csv"
)

VIDEO_INFO_FILE = os.path.join(
    BASE,
    "GAVD-main",
    "stage2_test",
    "video_info.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "GAVD-main",
    "stage2_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("GAVD ANNOTATION → ORIGINAL VIDEO ALIGNMENT AUDIT")
print("=" * 80)

print("\nLoading annotations...")

df = pd.read_csv(
    ANNOTATION_FILE,
    low_memory=False
)

print("Annotation rows:", len(df))

print("\nLoading video information...")

video_info = pd.read_csv(
    VIDEO_INFO_FILE
)

print("Downloaded videos inspected:", len(video_info))


# ============================================================
# CLEAN TYPES
# ============================================================

df["id"] = df["id"].astype(str)

df["seq"] = df["seq"].astype(str)

df["frame_num"] = pd.to_numeric(
    df["frame_num"],
    errors="coerce"
)

video_info["id"] = video_info["id"].astype(str)


# ============================================================
# ONLY VIDEOS WE SUCCESSFULLY DOWNLOADED
# ============================================================

available_ids = set(
    video_info["id"]
)

df_test = df[
    df["id"].isin(available_ids)
].copy()

print(
    "\nAnnotation rows belonging to downloaded videos:",
    len(df_test)
)


# ============================================================
# VIDEO-LEVEL ALIGNMENT
# ============================================================

video_results = []

for video_id, group in df_test.groupby("id"):

    info = video_info[
        video_info["id"] == video_id
    ]

    if len(info) == 0:
        continue

    info = info.iloc[0]

    fps = float(info["fps"])

    actual_video_frames = int(
        info["frame_count"]
    )

    video_duration = float(
        info["duration_seconds"]
    )

    min_frame = int(
        group["frame_num"].min()
    )

    max_frame = int(
        group["frame_num"].max()
    )

    unique_frames = (
        group["frame_num"]
        .dropna()
        .astype(int)
        .unique()
    )

    unique_frames = np.sort(
        unique_frames
    )

    annotation_frame_count = len(
        unique_frames
    )

    # --------------------------------------------------------
    # Missing frame numbers INSIDE annotation range
    # --------------------------------------------------------

    if len(unique_frames) > 1:

        expected_frames = np.arange(
            min_frame,
            max_frame + 1
        )

        missing_frames = np.setdiff1d(
            expected_frames,
            unique_frames
        )

        frame_steps = np.diff(
            unique_frames
        )

        median_step = np.median(
            frame_steps
        )

        min_step = np.min(
            frame_steps
        )

        max_step = np.max(
            frame_steps
        )

    else:

        missing_frames = np.array([])

        median_step = np.nan
        min_step = np.nan
        max_step = np.nan

    # --------------------------------------------------------
    # Convert annotation frame range to time
    # --------------------------------------------------------

    if fps > 0:

        annotation_start_time = (
            min_frame / fps
        )

        annotation_end_time = (
            max_frame / fps
        )

        annotation_duration = (
            (max_frame - min_frame) / fps
        )

    else:

        annotation_start_time = np.nan
        annotation_end_time = np.nan
        annotation_duration = np.nan

    # --------------------------------------------------------
    # Does annotation extend beyond video?
    # --------------------------------------------------------

    frames_beyond_video = (
        max_frame >= actual_video_frames
    )

    # --------------------------------------------------------
    # Sequence count
    # --------------------------------------------------------

    sequence_count = group["seq"].nunique()

    # --------------------------------------------------------
    # Camera views
    # --------------------------------------------------------

    camera_views = "; ".join(
        sorted(
            group["cam_view"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    # --------------------------------------------------------
    # Gait classes
    # --------------------------------------------------------

    gait_classes = "; ".join(
        sorted(
            group["gait_pat"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    video_results.append({

        "id": video_id,

        "gait_pat": gait_classes,

        "cam_view": camera_views,

        "fps": fps,

        "actual_video_frames":
            actual_video_frames,

        "video_duration_seconds":
            video_duration,

        "annotation_min_frame":
            min_frame,

        "annotation_max_frame":
            max_frame,

        "annotation_unique_frames":
            annotation_frame_count,

        "annotation_range_frames":
            max_frame - min_frame + 1,

        "missing_frames_inside_range":
            len(missing_frames),

        "median_frame_step":
            median_step,

        "min_frame_step":
            min_step,

        "max_frame_step":
            max_step,

        "annotation_start_seconds":
            annotation_start_time,

        "annotation_end_seconds":
            annotation_end_time,

        "annotation_duration_seconds":
            annotation_duration,

        "sequence_count":
            sequence_count,

        "frames_beyond_video":
            frames_beyond_video

    })


# ============================================================
# SAVE VIDEO-LEVEL RESULTS
# ============================================================

video_alignment = pd.DataFrame(
    video_results
)

video_output = os.path.join(
    OUTPUT_DIR,
    "annotation_video_alignment.csv"
)

video_alignment.to_csv(
    video_output,
    index=False
)


# ============================================================
# SEQUENCE-LEVEL ANALYSIS
# ============================================================

print("\nAnalyzing individual GAVD sequences...")

sequence_results = []

for (video_id, seq), group in df_test.groupby(
    ["id", "seq"]
):

    info = video_info[
        video_info["id"] == video_id
    ]

    if len(info) == 0:
        continue

    info = info.iloc[0]

    fps = float(info["fps"])

    actual_video_frames = int(
        info["frame_count"]
    )

    frames = (
        group["frame_num"]
        .dropna()
        .astype(int)
        .unique()
    )

    frames = np.sort(frames)

    if len(frames) == 0:
        continue

    start_frame = int(
        frames.min()
    )

    end_frame = int(
        frames.max()
    )

    number_of_unique_frames = len(
        frames
    )

    # --------------------------------------------------------
    # Frame gaps
    # --------------------------------------------------------

    if len(frames) > 1:

        steps = np.diff(frames)

        median_step = np.median(
            steps
        )

        min_step = np.min(
            steps
        )

        max_step = np.max(
            steps
        )

        expected = np.arange(
            start_frame,
            end_frame + 1
        )

        missing = np.setdiff1d(
            expected,
            frames
        )

        missing_count = len(
            missing
        )

    else:

        median_step = np.nan
        min_step = np.nan
        max_step = np.nan
        missing_count = 0

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if fps > 0:

        duration_seconds = (
            end_frame - start_frame
        ) / fps

    else:

        duration_seconds = np.nan

    # --------------------------------------------------------
    # Video boundary check
    # --------------------------------------------------------

    beyond_video = (
        end_frame >= actual_video_frames
    )

    # --------------------------------------------------------
    # Gait event count
    # --------------------------------------------------------

    gait_events = group[
        "gait_event"
    ].notna().sum()

    sequence_results.append({

        "id": video_id,

        "seq": seq,

        "gait_pat":
            group["gait_pat"].iloc[0],

        "cam_view":
            group["cam_view"].iloc[0],

        "fps": fps,

        "start_frame":
            start_frame,

        "end_frame":
            end_frame,

        "unique_annotation_frames":
            number_of_unique_frames,

        "sequence_range_frames":
            end_frame - start_frame + 1,

        "duration_seconds":
            duration_seconds,

        "median_frame_step":
            median_step,

        "min_frame_step":
            min_step,

        "max_frame_step":
            max_step,

        "missing_frames":
            missing_count,

        "gait_event_rows":
            gait_events,

        "extends_beyond_video":
            beyond_video

    })


sequence_alignment = pd.DataFrame(
    sequence_results
)

sequence_output = os.path.join(
    OUTPUT_DIR,
    "annotation_sequence_alignment.csv"
)

sequence_alignment.to_csv(
    sequence_output,
    index=False
)


# ============================================================
# PRINT VIDEO-LEVEL RESULTS
# ============================================================

print()
print("=" * 80)
print("VIDEO-LEVEL ALIGNMENT")
print("=" * 80)

print(
    video_alignment.to_string(
        index=False
    )
)


# ============================================================
# PRINT SEQUENCE STATISTICS
# ============================================================

print()
print("=" * 80)
print("SEQUENCE-LEVEL STATISTICS")
print("=" * 80)

if len(sequence_alignment) > 0:

    print(
        sequence_alignment[
            "duration_seconds"
        ].describe().to_string()
    )

    print()

    print(
        "Total sequences:",
        len(sequence_alignment)
    )

    print(
        "Sequences extending beyond video:",
        sequence_alignment[
            "extends_beyond_video"
        ].sum()
    )

    print(
        "Sequences containing gait events:",
        (
            sequence_alignment[
                "gait_event_rows"
            ] > 0
        ).sum()
    )

    print()
    print("Sequence duration distribution:")
    print(
        sequence_alignment[
            "duration_seconds"
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .to_string()
    )


# ============================================================
# FPS DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("FPS DISTRIBUTION")
print("=" * 80)

print(
    video_alignment[
        "fps"
    ].value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SAVE MISSING FRAME DETAILS
# ============================================================

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    video_output
)

print(
    sequence_output
)

print()
print("Audit complete.")