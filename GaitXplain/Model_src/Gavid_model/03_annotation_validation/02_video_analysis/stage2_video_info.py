import cv2
import pandas as pd
import os

# ============================================================
# PATHS
# ============================================================

BASE = "/media/legion/Data/Final Year Project/New preprocessing"

CSV_FILE = os.path.join(
    BASE,
    "GAVD-main",
    "stage2_test",
    "test_videos.csv"
)

VIDEO_DIR = os.path.join(
    BASE,
    "GAVD-main",
    "stage2_test",
    "videos"
)

OUTPUT = os.path.join(
    BASE,
    "GAVD-main",
    "stage2_test",
    "video_info.csv"
)

# ============================================================
# READ TEST VIDEO LIST
# ============================================================

df = pd.read_csv(CSV_FILE)

results = []

print("=" * 80)
print("GAVD ORIGINAL VIDEO INSPECTION")
print("=" * 80)

# ============================================================
# INSPECT EACH VIDEO
# ============================================================

for _, row in df.iterrows():

    video_id = str(row["id"])

    video_path = os.path.join(
        VIDEO_DIR,
        video_id + ".mp4"
    )

    # --------------------------------------------------------
    # Check if video exists
    # --------------------------------------------------------

    if not os.path.exists(video_path):

        print(f"MISSING : {video_id}")
        continue

    print(f"Checking: {video_id}")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(f"FAILED  : {video_id}")
        continue

    # --------------------------------------------------------
    # Get video information
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # --------------------------------------------------------
    # Calculate duration
    # --------------------------------------------------------

    if fps > 0:

        duration = frame_count / fps

    else:

        duration = 0

    # --------------------------------------------------------
    # Release video
    # --------------------------------------------------------

    cap.release()

    # --------------------------------------------------------
    # Store information
    # --------------------------------------------------------

    results.append({

        "id": video_id,

        "gait_pat": row["gait_pat"],

        "cam_view": row["cam_view"],

        # Original video information
        "fps": fps,

        "frame_count": frame_count,

        "duration_seconds": duration,

        "width": width,

        "height": height,

        # Information from GAVD annotations
        "annotation_frames": row["frame_count"],

        "annotation_sequences": row["sequence_count"]

    })

# ============================================================
# CREATE DATAFRAME
# ============================================================

info = pd.DataFrame(results)

# ============================================================
# SAVE RESULTS
# ============================================================

info.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 80)
print("VIDEO INFORMATION")
print("=" * 80)

if len(info) > 0:

    print(
        info.to_string(index=False)
    )

else:

    print("No videos could be inspected.")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Videos successfully inspected:",
    len(info)
)

if len(info) > 0:

    print()
    print("FPS distribution:")
    print(
        info["fps"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Duration statistics (seconds):")
    print(
        info["duration_seconds"]
        .describe()
        .to_string()
    )

    print()
    print("Frame-count statistics:")
    print(
        info["frame_count"]
        .describe()
        .to_string()
    )

    print()
    print("Resolutions:")
    print(
        info[["width", "height"]]
        .value_counts()
        .to_string()
    )

# ============================================================
# OUTPUT LOCATION
# ============================================================

print()
print("=" * 80)
print("Saved results to:")
print(OUTPUT)
print("=" * 80)