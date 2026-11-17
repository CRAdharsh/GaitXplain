import os
import csv
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ============================================================
# CONFIG
# ============================================================

BASE = "/Final Year Project/GAVD-main"

VIDEO_DIR = os.path.join(BASE, "stage2_test", "videos")
ALIGNMENT_CSV = os.path.join(
    BASE, "stage2_test", "annotation_sequence_alignment.csv"
)
TEST_CSV = os.path.join(
    BASE, "stage2_test", "test_videos.csv"
)
V2_DIR = os.path.join(
    BASE, "stage2_test", "v2_2sec"
)
NPZ_DIR = os.path.join(V2_DIR, "sequences")

# The videos that produced ZERO windows
PROBLEM_VIDEOS = [
    "KnvYRnTA3XQ",
    "R8LRCiTvUz8",
    "TxqRbz0bh94",
    "_cJ8RK8-Emw",
    "-oJM2wUUjws",
    "7xcj-byGS4w",
    "H7WThdv-fCQ",
]

WINDOW_SECONDS = 2.0
TOKEN_COUNT = 32

# ============================================================
# HELPERS
# ============================================================

def find_video(video_id):
    extensions = [".mp4", ".webm", ".mkv", ".mov", ".avi"]

    for ext in extensions:
        path = os.path.join(VIDEO_DIR, video_id + ext)
        if os.path.exists(path):
            return path

    # fallback: search by ID
    for f in os.listdir(VIDEO_DIR):
        if f.startswith(video_id):
            return os.path.join(VIDEO_DIR, f)

    return None


def safe_int(x):
    try:
        return int(float(x))
    except:
        return None


def print_separator():
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print_separator()
print("GAVD V2 — ZERO-OUTPUT VIDEO DIAGNOSTIC")
print_separator()

print("\nLoading annotation data...")

alignment = pd.read_csv(ALIGNMENT_CSV)
test_videos = pd.read_csv(TEST_CSV)

print(f"Alignment rows : {len(alignment)}")
print(f"Test videos    : {len(test_videos)}")

# ============================================================
# PART 1 — NPZ TEMPORAL CONTENT
# ============================================================

print("\n")
print_separator()
print("PART 1 — NPZ TEMPORAL ARRAY CHECK")
print_separator()

npz_files = []

if os.path.isdir(NPZ_DIR):
    npz_files = [
        os.path.join(NPZ_DIR, f)
        for f in os.listdir(NPZ_DIR)
        if f.endswith(".npz")
    ]

print(f"NPZ files found: {len(npz_files)}")

if npz_files:

    sample = npz_files[0]

    print(f"\nChecking sample:")
    print(sample)

    try:
        data = np.load(sample)

        print("\nArrays inside NPZ:")

        for key in data.files:
            arr = data[key]
            print(
                f"  {key:25s} "
                f"shape={str(arr.shape):20s} "
                f"dtype={arr.dtype}"
            )

        required_temporal = [
            "frame_numbers",
            "timestamps",
            "relative_timestamps",
            "delta_t",
            "source_fps",
        ]

        print("\nTemporal arrays:")

        for key in required_temporal:
            if key in data.files:
                print(f"  [OK]      {key}")
            else:
                print(f"  [MISSING] {key}")

    except Exception as e:
        print(f"ERROR reading NPZ: {e}")


# ============================================================
# PART 2 — VIDEO DIAGNOSTICS
# ============================================================

print("\n")
print_separator()
print("PART 2 — VIDEO DECODING DIAGNOSTICS")
print_separator()

results = []

for video_id in PROBLEM_VIDEOS:

    print("\n")
    print("-" * 80)
    print(f"VIDEO: {video_id}")
    print("-" * 80)

    video_path = find_video(video_id)

    if video_path is None:

        print("[NOT FOUND]")
        print("No local video file exists.")

        results.append({
            "video_id": video_id,
            "video_found": False,
            "opencv_open": False,
            "total_frames_reported": 0,
            "decoded_frames": 0,
            "decode_failures": 0,
            "annotated_frames": 0,
            "annotated_readable": 0,
            "annotated_unreadable": 0,
        })

        continue

    print(f"Video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    opened = cap.isOpened()

    print(f"OpenCV opened: {opened}")

    if not opened:

        results.append({
            "video_id": video_id,
            "video_found": True,
            "opencv_open": False,
            "total_frames_reported": 0,
            "decoded_frames": 0,
            "decode_failures": 0,
            "annotated_frames": 0,
            "annotated_readable": 0,
            "annotated_unreadable": 0,
        })

        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS             : {fps}")
    print(f"Reported frames : {frame_count}")
    print(f"Resolution      : {width} x {height}")

    # --------------------------------------------------------
    # Annotation information
    # --------------------------------------------------------

    video_ann = alignment[
        alignment["id"].astype(str) == str(video_id)
    ]

    if len(video_ann) == 0:

        print("[NO ALIGNMENT ROWS]")

        cap.release()

        results.append({
            "video_id": video_id,
            "video_found": True,
            "opencv_open": True,
            "total_frames_reported": frame_count,
            "decoded_frames": 0,
            "decode_failures": 0,
            "annotated_frames": 0,
            "annotated_readable": 0,
            "annotated_unreadable": 0,
        })

        continue

    annotated_frames = set()

    for _, row in video_ann.iterrows():

        start = safe_int(row["start_frame"])
        end = safe_int(row["end_frame"])

        if start is None or end is None:
            continue

        for frame_no in range(start, end + 1):
            annotated_frames.add(frame_no)

    annotated_frames = sorted(annotated_frames)

    print(f"Annotated unique frames: {len(annotated_frames)}")

    if annotated_frames:

        print(
            f"Annotation frame range : "
            f"{min(annotated_frames)} -> {max(annotated_frames)}"
        )

    # --------------------------------------------------------
    # Sequential decode
    # --------------------------------------------------------

    decoded = 0
    decode_failures = 0

    annotated_readable = 0
    annotated_unreadable = 0

    readable_annotation_frames = []
    unreadable_annotation_frames = []

    failure_ranges = []

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            # Determine whether we simply reached EOF
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame_index < frame_count - 1:
                decode_failures += 1

            break

        decoded += 1

        # GAVD frame numbers are 1-based
        gavd_frame = frame_index + 1

        if gavd_frame in annotated_frames:

            if frame is None or frame.size == 0:

                annotated_unreadable += 1
                unreadable_annotation_frames.append(gavd_frame)

            else:

                annotated_readable += 1
                readable_annotation_frames.append(gavd_frame)

        frame_index += 1

        # Do not unnecessarily decode massive videos forever.
        # We only need to know whether the annotation range
        # can be reached.
        if (
            annotated_frames
            and frame_index >= max(annotated_frames)
        ):
            break

    cap.release()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nDECODING RESULT")

    print(f"Frames successfully decoded : {decoded}")
    print(f"Decode failures             : {decode_failures}")

    print("\nANNOTATION READABILITY")

    print(
        f"Readable annotated frames   : "
        f"{annotated_readable}"
    )

    print(
        f"Unreadable annotated frames : "
        f"{annotated_unreadable}"
    )

    if annotated_frames:

        coverage = (
            annotated_readable / len(annotated_frames)
        ) * 100

        print(
            f"Annotation decode coverage  : "
            f"{coverage:.2f}%"
        )

    # --------------------------------------------------------
    # First/last readable frame
    # --------------------------------------------------------

    if readable_annotation_frames:

        print(
            f"\nFirst readable annotation frame: "
            f"{min(readable_annotation_frames)}"
        )

        print(
            f"Last readable annotation frame : "
            f"{max(readable_annotation_frames)}"
        )

    # --------------------------------------------------------
    # Unreadable frame sample
    # --------------------------------------------------------

    if unreadable_annotation_frames:

        print("\nUnreadable annotation frame sample:")

        print(
            unreadable_annotation_frames[:30]
        )

    results.append({
        "video_id": video_id,
        "video_found": True,
        "opencv_open": opened,
        "total_frames_reported": frame_count,
        "decoded_frames": decoded,
        "decode_failures": decode_failures,
        "annotated_frames": len(annotated_frames),
        "annotated_readable": annotated_readable,
        "annotated_unreadable": annotated_unreadable,
    })


# ============================================================
# PART 3 — SEQUENCE WINDOW FEASIBILITY
# ============================================================

print("\n")
print_separator()
print("PART 3 — 2-SECOND WINDOW FEASIBILITY")
print_separator()

print(
    "\nThis section checks whether the annotation duration "
    "should theoretically allow 2-second windows."
)

sequence_results = []

for video_id in PROBLEM_VIDEOS:

    video_ann = alignment[
        alignment["id"].astype(str) == str(video_id)
    ]

    if len(video_ann) == 0:
        continue

    print("\n")
    print(f"VIDEO: {video_id}")

    for _, row in video_ann.iterrows():

        seq = row["seq"]

        fps = float(row["fps"])

        start = safe_int(row["start_frame"])
        end = safe_int(row["end_frame"])

        n_frames = int(row["unique_annotation_frames"])

        duration = float(row["duration_seconds"])

        # Approximate number of 2-sec windows
        if duration < WINDOW_SECONDS:
            expected = 0
        else:
            step = WINDOW_SECONDS * (1 - 0.5)

            expected = (
                int(
                    np.floor(
                        (duration - WINDOW_SECONDS)
                        / step
                    )
                ) + 1
            )

        possible = (
            duration >= WINDOW_SECONDS
            and n_frames >= 2
        )

        status = "SHOULD HAVE WINDOWS" if expected > 0 else "TOO SHORT"

        print(
            f"  {seq} | "
            f"{duration:7.3f}s | "
            f"{n_frames:4d} frames | "
            f"expected={expected:3d} | "
            f"{status}"
        )

        sequence_results.append({
            "video_id": video_id,
            "sequence_id": seq,
            "fps": fps,
            "start_frame": start,
            "end_frame": end,
            "annotated_frames": n_frames,
            "duration_seconds": duration,
            "expected_windows": expected,
        })


# ============================================================
# PART 4 — FINAL DIAGNOSIS
# ============================================================

print("\n")
print_separator()
print("PART 4 — DIAGNOSIS")
print_separator()

df = pd.DataFrame(results)

if len(df):

    print("\nVIDEO SUMMARY")
    print(df.to_string(index=False))

print("\n\nINTERPRETATION")

for _, row in df.iterrows():

    vid = row["video_id"]

    if not row["video_found"]:
        diagnosis = "VIDEO FILE MISSING"

    elif not row["opencv_open"]:
        diagnosis = "OPENCV CANNOT OPEN VIDEO"

    elif row["annotated_frames"] == 0:
        diagnosis = "NO ANNOTATED FRAMES FOUND"

    elif row["annotated_readable"] == 0:
        diagnosis = (
            "CRITICAL: VIDEO OPENS BUT NONE OF THE "
            "ANNOTATED FRAMES CAN BE DECODED"
        )

    elif row["annotated_unreadable"] > 0:
        diagnosis = (
            "PARTIAL DECODING FAILURE IN ANNOTATED FRAMES"
        )

    else:
        diagnosis = (
            "ANNOTATED FRAMES ARE DECODABLE — "
            "INVESTIGATE FEATURE EXTRACTION / WINDOW LOGIC"
        )

    print(f"\n{vid}")
    print(f"  -> {diagnosis}")


# ============================================================
# SAVE RESULTS
# ============================================================

output_csv = os.path.join(
    V2_DIR,
    "diagnostic_video_results.csv"
)

df.to_csv(output_csv, index=False)

seq_csv = os.path.join(
    V2_DIR,
    "diagnostic_sequence_results.csv"
)

pd.DataFrame(sequence_results).to_csv(
    seq_csv,
    index=False
)

print("\n")
print_separator()
print("DIAGNOSTIC COMPLETE")
print_separator()

print(f"\nVideo report:")
print(output_csv)

print(f"\nSequence report:")
print(seq_csv)

print("\nNO DATA WAS MODIFIED.")