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
    "sampling_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("GAVD TEMPORAL SAMPLING ANALYSIS")
print("=" * 70)

print(f"\nTotal annotated sequences: {len(df)}")


# ============================================================
# BASIC SEQUENCE INFORMATION
# ============================================================

# Number of actual annotated frames
df["frame_count"] = df["unique_annotation_frames"]

# Duration calculated from actual annotated frame count
df["duration_calc"] = (
    df["frame_count"] / df["fps"]
)


# ============================================================
# 1. SEQUENCE LENGTH DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. SEQUENCE LENGTH DISTRIBUTION")
print("=" * 70)

bins = [
    0,
    32,
    64,
    128,
    256,
    float("inf")
]

labels = [
    "<32",
    "32-63",
    "64-127",
    "128-255",
    "256+"
]

df["length_group"] = pd.cut(
    df["frame_count"],
    bins=bins,
    labels=labels,
    right=False
)

length_counts = (
    df["length_group"]
    .value_counts()
    .reindex(labels)
    .fillna(0)
    .astype(int)
)

for label, count in length_counts.items():

    percentage = count / len(df) * 100

    print(
        f"{label:10s}: "
        f"{count:5d} sequences "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 2. 32-FRAME FEASIBILITY
# ============================================================

print("\n" + "=" * 70)
print("2. 32-FRAME FEASIBILITY")
print("=" * 70)

total = len(df)

can_32 = (
    df["frame_count"] >= 32
).sum()

cannot_32 = (
    df["frame_count"] < 32
).sum()

print(
    f"Sequences >= 32 frames : {can_32}"
)

print(
    f"Sequences < 32 frames  : {cannot_32}"
)

print(
    f"Usable percentage      : "
    f"{can_32 / total * 100:.2f}%"
)


# ============================================================
# 3. WINDOW CALCULATOR
# ============================================================

def count_windows(
    frame_count,
    window=32,
    stride=32
):

    if frame_count < window:
        return 0

    return (
        (frame_count - window) // stride
    ) + 1


# ============================================================
# 4. NON-OVERLAPPING WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("3. NON-OVERLAPPING 32-FRAME WINDOWS")
print("=" * 70)

df["windows_stride_32"] = (
    df["frame_count"]
    .apply(
        lambda x:
        count_windows(
            x,
            32,
            32
        )
    )
)

print(
    "Total samples with stride 32:",
    int(df["windows_stride_32"].sum())
)


# ============================================================
# 5. OVERLAPPING WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("4. OVERLAPPING 32-FRAME WINDOWS")
print("=" * 70)

for stride in [16, 8, 4]:

    column = f"windows_stride_{stride}"

    df[column] = (
        df["frame_count"]
        .apply(
            lambda x:
            count_windows(
                x,
                32,
                stride
            )
        )
    )

    total_windows = int(
        df[column].sum()
    )

    print(
        f"Stride {stride:2d}: "
        f"{total_windows:7d} samples"
    )


# ============================================================
# 6. UNIFORM 32-FRAME REPRESENTATION
# ============================================================

print("\n" + "=" * 70)
print("5. UNIFORM 32-FRAME REPRESENTATION")
print("=" * 70)

print(
    """
For sequences >=32 frames:

We can select 32 actual frames distributed
across the complete annotated sequence.

No interpolation is required.

Example:

100-frame sequence
       |
       v
32 actual frames selected across it.
"""
)

df["uniform_32_possible"] = (
    df["frame_count"] >= 32
)

uniform_count = int(
    df["uniform_32_possible"].sum()
)

print(
    f"Sequences representable with "
    f"uniform 32 frames: {uniform_count}"
)


# ============================================================
# 7. TEMPORAL DURATION
# ============================================================

print("\n" + "=" * 70)
print("6. SEQUENCE DURATION")
print("=" * 70)

duration = df["duration_seconds"]

print(
    f"Minimum : {duration.min():.2f} sec"
)

print(
    f"Mean    : {duration.mean():.2f} sec"
)

print(
    f"Median  : {duration.median():.2f} sec"
)

print(
    f"75%     : {duration.quantile(0.75):.2f} sec"
)

print(
    f"Maximum : {duration.max():.2f} sec"
)


# ============================================================
# 8. DURATION GROUPS
# ============================================================

print("\n" + "=" * 70)
print("7. DURATION GROUPS")
print("=" * 70)

duration_bins = [
    0,
    2,
    4,
    6,
    8,
    10,
    15,
    30,
    float("inf")
]

duration_labels = [
    "<2s",
    "2-4s",
    "4-6s",
    "6-8s",
    "8-10s",
    "10-15s",
    "15-30s",
    "30s+"
]

df["duration_group"] = pd.cut(
    duration,
    bins=duration_bins,
    labels=duration_labels,
    right=False
)

duration_counts = (
    df["duration_group"]
    .value_counts()
    .reindex(duration_labels)
    .fillna(0)
    .astype(int)
)

for label, count in duration_counts.items():

    print(
        f"{label:8s}: {count:5d}"
    )


# ============================================================
# 9. CLASS-LEVEL SAMPLE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("8. CLASS-LEVEL SAMPLE COUNTS")
print("=" * 70)

class_summary = (
    df.groupby(
        "gait_pat",
        observed=True
    )
    .agg(
        sequences=("seq", "count"),
        stride32=("windows_stride_32", "sum"),
        stride16=("windows_stride_16", "sum"),
        stride8=("windows_stride_8", "sum"),
        stride4=("windows_stride_4", "sum"),
    )
    .sort_index()
)

print(
    class_summary.to_string()
)


# ============================================================
# 10. FPS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("9. FPS DISTRIBUTION")
print("=" * 70)

fps_summary = (
    df.groupby("fps")
    .size()
    .sort_index()
)

for fps, count in fps_summary.items():

    print(
        f"{fps:10.3f} FPS : "
        f"{count:5d} sequences"
    )


# ============================================================
# 11. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. DATA QUALITY CHECK")
print("=" * 70)

print(
    "Sequences with missing annotated frames:",
    int((df["missing_frames"] > 0).sum())
)

print(
    "Sequences with non-continuous frame steps:",
    int((df["median_frame_step"] != 1).sum())
)

print(
    "Sequences flagged beyond video:",
    int(df["extends_beyond_video"].sum())
)


# ============================================================
# 12. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nDetailed results saved to:\n"
    f"{OUTPUT_FILE}"
)