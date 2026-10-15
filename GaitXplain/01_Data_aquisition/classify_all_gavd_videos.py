import os
import re
import subprocess
import pandas as pd
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/Final Year Project/GAVD-main"

DATA_DIR = os.path.join(BASE_DIR, "data")

FULL_DIR = os.path.join(BASE_DIR, "full_dataset")
VIDEO_DIR = os.path.join(FULL_DIR, "videos")
STATUS_FILE = os.path.join(FULL_DIR, "download_status.csv")

OUTPUT_FILE = os.path.join(FULL_DIR, "availability_classification.csv")
LOG_DIR = os.path.join(FULL_DIR, "availability_logs")

os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

YT_DLP = "python"

TIMEOUT_SECONDS = 60

# ============================================================
# LOAD ALL GAVD ANNOTATIONS
# ============================================================

print("=" * 70)
print("GAVD VIDEO AVAILABILITY CLASSIFIER")
print("=" * 70)

csv_files = [
    os.path.join(DATA_DIR, f"GAVD_Clinical_Annotations_{i}.csv")
    for i in range(1, 6)
]

all_annotations = []

for csv_file in csv_files:
    if not os.path.exists(csv_file):
        print(f"[WARNING] Missing: {csv_file}")
        continue

    print(f"Reading: {os.path.basename(csv_file)}")

    df = pd.read_csv(csv_file)

    if "id" not in df.columns:
        print(f"[WARNING] No 'id' column in {csv_file}")
        continue

    all_annotations.append(df)

if not all_annotations:
    raise RuntimeError("No annotation CSV files found.")

annotations = pd.concat(all_annotations, ignore_index=True)

# ============================================================
# CREATE MASTER VIDEO LIST
# ============================================================

video_columns = [
    "id",
    "gait_pat",
    "cam_view",
    "url"
]

available_columns = [
    c for c in video_columns
    if c in annotations.columns
]

videos = (
    annotations[available_columns]
    .drop_duplicates(subset=["id"])
    .copy()
)

videos = videos.sort_values("id").reset_index(drop=True)

print()
print(f"Total unique videos from GAVD: {len(videos)}")

# ============================================================
# LOAD EXISTING DOWNLOAD STATUS
# ============================================================

if os.path.exists(STATUS_FILE):

    old_status = pd.read_csv(STATUS_FILE)

    print(f"Existing status entries: {len(old_status)}")

    # Keep only useful columns if they exist
    keep_cols = [
        "id",
        "status",
        "error",
        "filename",
        "timestamp"
    ]

    keep_cols = [
        c for c in keep_cols
        if c in old_status.columns
    ]

    old_status = old_status[keep_cols].copy()

else:

    print("No existing download_status.csv found.")

    old_status = pd.DataFrame(
        columns=[
            "id",
            "status",
            "error",
            "filename",
            "timestamp"
        ]
    )

# ============================================================
# MERGE
# ============================================================

if "id" in old_status.columns:

    # Rename old status so it doesn't collide
    old_status = old_status.rename(
        columns={
            "status": "download_status",
            "error": "download_error",
            "filename": "download_filename",
            "timestamp": "download_timestamp"
        }
    )

    videos = videos.merge(
        old_status,
        on="id",
        how="left"
    )

else:

    videos["download_status"] = None
    videos["download_error"] = None
    videos["download_filename"] = None
    videos["download_timestamp"] = None


# ============================================================
# HELPER: FIND LOCAL VIDEO
# ============================================================

def find_local_video(video_id):

    possible_extensions = [
        ".mp4",
        ".mkv",
        ".webm",
        ".mov"
    ]

    for ext in possible_extensions:

        path = os.path.join(
            VIDEO_DIR,
            video_id + ext
        )

        if os.path.isfile(path):
            return path

    # Also check any file beginning with video ID
    if os.path.isdir(VIDEO_DIR):

        for filename in os.listdir(VIDEO_DIR):

            if filename.startswith(video_id):

                path = os.path.join(
                    VIDEO_DIR,
                    filename
                )

                if os.path.isfile(path):
                    return path

    return None


# ============================================================
# HELPER: CLASSIFY YT-DLP OUTPUT
# ============================================================

def classify_output(output, return_code):

    text = output.lower()

    # --------------------------------------------------------
    # PRIVATE
    # --------------------------------------------------------

    if (
        "private video" in text
        or "this video is private" in text
    ):
        return "private"

    # --------------------------------------------------------
    # REMOVED
    # --------------------------------------------------------

    if (
        "video has been removed" in text
        or "removed by the uploader" in text
        or "video was removed" in text
    ):
        return "removed"

    # --------------------------------------------------------
    # MEMBERS ONLY
    # --------------------------------------------------------

    if (
        "members-only" in text
        or "members only" in text
        or "join this channel" in text
    ):
        return "members_only"

    # --------------------------------------------------------
    # LOGIN / AUTHENTICATION
    # --------------------------------------------------------

    if (
        "sign in to confirm" in text
        or "login required" in text
        or "authentication required" in text
        or "sign in" in text
        and "confirm" in text
    ):
        return "login_required"

    # --------------------------------------------------------
    # UNAVAILABLE
    # --------------------------------------------------------

    if (
        "this video is unavailable" in text
        or "this video is not available" in text
        or "video unavailable" in text
        or "playability status" in text
        and "unplayable" in text
        or "unplayable" in text
    ):
        return "unavailable"

    # --------------------------------------------------------
    # AGE RESTRICTION
    # --------------------------------------------------------

    if (
        "age-restricted" in text
        or "age restricted" in text
        or "confirm your age" in text
    ):
        return "age_restricted"

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if return_code == 0:

        # If yt-dlp successfully returned video information
        return "available"

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    return "other_error"


# ============================================================
# CHECK ONE VIDEO
# ============================================================

def check_video(video_id, url):

    print()
    print("-" * 70)
    print(f"Checking: {video_id}")
    print(f"URL: {url}")

    command = [
        YT_DLP,
        "-m",
        "yt_dlp",

        "--cookies-from-browser",
        "firefox",

        "--no-playlist",

        "--skip-download",

        "-F",

        "--socket-timeout",
        "30",

        url
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        output = result.stdout

        classification = classify_output(
            output,
            result.returncode
        )

        print(f"Result: {classification}")

        return classification, output

    except subprocess.TimeoutExpired as e:

        output = ""

        if e.stdout:
            try:
                output = e.stdout.decode(
                    errors="ignore"
                )
            except Exception:
                output = str(e.stdout)

        print("Result: timeout")

        return "timeout", output

    except Exception as e:

        print(f"Result: error ({e})")

        return "other_error", str(e)


# ============================================================
# PROCESS ALL 348 VIDEOS
# ============================================================

results = []

total = len(videos)

for index, row in videos.iterrows():

    video_id = str(row["id"])

    url = str(row["url"])

    local_video = find_local_video(video_id)

    # --------------------------------------------------------
    # LOCAL VIDEO EXISTS
    # --------------------------------------------------------

    if local_video:

        classification = "available"

        output = "Local video file exists."

        print()
        print(
            f"[{index + 1}/{total}] "
            f"{video_id} -> available (local file)"
        )

    else:

        # ----------------------------------------------------
        # CHECK ONLINE
        # ----------------------------------------------------

        print()
        print(
            f"[{index + 1}/{total}] "
            f"{video_id}"
        )

        classification, output = check_video(
            video_id,
            url
        )

    # --------------------------------------------------------
    # SAVE RAW LOG
    # --------------------------------------------------------

    log_file = os.path.join(
        LOG_DIR,
        f"{video_id}.txt"
    )

    try:

        with open(
            log_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(output)

    except Exception:
        pass

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append({
        "id": video_id,
        "gait_pat": row.get("gait_pat", ""),
        "cam_view": row.get("cam_view", ""),
        "url": url,
        "classification": classification,
        "local_file": local_video if local_video else "",
        "checked_at": datetime.now().isoformat()
    })

    # --------------------------------------------------------
    # SAVE PROGRESS AFTER EVERY VIDEO
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

result_df = pd.DataFrame(results)

print()
print()
print("=" * 70)
print("FINAL CLASSIFICATION")
print("=" * 70)

print(
    result_df["classification"]
    .value_counts(dropna=False)
)

print()
print(f"Total videos checked: {len(result_df)}")
print(f"Output: {OUTPUT_FILE}")
print(f"Logs:   {LOG_DIR}")

# ============================================================
# SHOW UNAVAILABLE / PRIVATE / OTHER
# ============================================================

print()
print("=" * 70)
print("UNAVAILABLE")
print("=" * 70)

print(
    result_df[
        result_df["classification"] == "unavailable"
    ][
        ["id", "gait_pat", "cam_view"]
    ].to_string(index=False)
)

print()
print("=" * 70)
print("PRIVATE")
print("=" * 70)

print(
    result_df[
        result_df["classification"] == "private"
    ][
        ["id", "gait_pat", "cam_view"]
    ].to_string(index=False)
)

print()
print("=" * 70)
print("OTHER ERRORS")
print("=" * 70)

print(
    result_df[
        result_df["classification"] == "other_error"
    ][
        ["id", "gait_pat", "cam_view"]
    ].to_string(index=False)
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)