import pandas as pd
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

BASE = "/media/legion/Data/Final Year Project/New preprocessing/GAVD-main/stage2_test"

INPUT_FILE = os.path.join(
    BASE,
    "annotation_sequence_alignment.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "time_window_analysis.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_DURATIONS = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0
]

# Temporal overlap.
#
# 0.50 means the next window starts halfway
# through the previous window.
#
# Example for 2-second window:
#
# Window 1: 0.0 - 2.0
# Window 2: 1.0 - 3.0
# Window 3: 2.0 - 4.0
#
OVERLAP = 0.50


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 75)
print("GAVD FIXED-TIME WINDOW ANALYSIS")
print("=" * 75)

print(
    f"\nTotal annotated sequences: {len(df)}"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

df["frame_count"] = (
    df["unique_annotation_frames"]
)

df["duration"] = (
    df["duration_seconds"]
)

df["fps"] = pd.to_numeric(
    df["fps"],
    errors="coerce"
)


# ============================================================
# WINDOW CALCULATOR
# ============================================================

def calculate_windows(
    sequence_duration,
    fps,
    window_duration,
    overlap
):

    if pd.isna(sequence_duration):
        return 0

    if pd.isna(fps):
        return 0

    if sequence_duration < window_duration:
        return 0

    # Time between starts of consecutive windows
    stride_duration = (
        window_duration *
        (1.0 - overlap)
    )

    # Number of complete windows
    windows = (
        int(
            np.floor(
                (
                    sequence_duration
                    - window_duration
                )
                / stride_duration
            )
        )
        + 1
    )

    return max(windows, 0)


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_results = []


for window_duration in WINDOW_DURATIONS:

    print("\n" + "=" * 75)
    print(
        f"WINDOW: {window_duration:.1f} SECONDS"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Number of windows per sequence
    # --------------------------------------------------------

    column_name = (
        f"windows_{window_duration:.1f}s"
    )

    df[column_name] = df.apply(
        lambda row:
        calculate_windows(
            row["duration"],
            row["fps"],
            window_duration,
            OVERLAP
        ),
        axis=1
    )

    # --------------------------------------------------------
    # Basic totals
    # --------------------------------------------------------

    total_sequences = len(df)

    usable_sequences = (
        df[column_name] > 0
    ).sum()

    total_windows = int(
        df[column_name].sum()
    )

    usable_percentage = (
        usable_sequences /
        total_sequences *
        100
    )

    average_windows = (
        total_windows /
        total_sequences
    )

    average_usable_windows = (
        total_windows /
        usable_sequences
        if usable_sequences > 0
        else 0
    )

    print(
        f"\nSequences able to produce "
        f"at least one window: "
        f"{usable_sequences}/{total_sequences}"
    )

    print(
        f"Usable sequences: "
        f"{usable_percentage:.2f}%"
    )

    print(
        f"Total windows: "
        f"{total_windows}"
    )

    print(
        f"Average windows / sequence: "
        f"{average_windows:.2f}"
    )

    print(
        f"Average windows / usable sequence: "
        f"{average_usable_windows:.2f}"
    )


    # ========================================================
    # WINDOWS BY CLASS
    # ========================================================

    print("\nWindows by class:")

    class_summary = (
        df.groupby(
            "gait_pat",
            observed=True
        )[column_name]
        .agg(
            sequences="count",
            usable_sequences=lambda x:
                (x > 0).sum(),
            windows="sum"
        )
        .sort_values(
            "windows",
            ascending=False
        )
    )

    print(
        class_summary.to_string()
    )


    # ========================================================
    # WINDOWS BY FPS
    # ========================================================

    print("\nWindows by FPS:")

    fps_summary = (
        df.groupby("fps")[column_name]
        .agg(
            sequences="count",
            usable_sequences=lambda x:
                (x > 0).sum(),
            windows="sum"
        )
        .sort_index()
    )

    print(
        fps_summary.to_string()
    )


    # ========================================================
    # STORE OVERALL RESULT
    # ========================================================

    all_results.append({

        "window_seconds":
            window_duration,

        "total_sequences":
            total_sequences,

        "usable_sequences":
            usable_sequences,

        "usable_percentage":
            usable_percentage,

        "total_windows":
            total_windows,

        "average_windows_per_sequence":
            average_windows,

        "average_windows_per_usable_sequence":
            average_usable_windows
    })


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 75)
print("OVERALL COMPARISON")
print("=" * 75)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# GPU / TEMPORAL TOKEN ESTIMATE
# ============================================================

print("\n")
print("=" * 75)
print("APPROXIMATE ORIGINAL FRAME COUNT")
print("=" * 75)

print(
    """
This shows approximately how many ORIGINAL
video frames exist inside each time window.

It is NOT the final Transformer token count.

The purpose is to understand the computational
difference between 24, 30 and 60 FPS videos.
"""
)

for duration in WINDOW_DURATIONS:

    print(
        f"\n{duration:.1f} second window:"
    )

    for fps in [23.976, 30.0, 60.0]:

        frames = duration * fps

        print(
            f"  {fps:6.3f} FPS -> "
            f"~{frames:.1f} original frames"
        )


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    f"\nSummary saved to:\n"
    f"{OUTPUT_FILE}"
)