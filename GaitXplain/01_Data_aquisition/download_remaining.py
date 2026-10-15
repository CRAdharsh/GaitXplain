import os
import subprocess
import pandas as pd
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/Final Year Project/GAVD-main"

FULL_DIR = os.path.join(BASE_DIR, "full_dataset")

VIDEO_DIR = os.path.join(FULL_DIR, "videos")
LOG_DIR = os.path.join(FULL_DIR, "download_logs")

AVAILABILITY_FILE = os.path.join(
    FULL_DIR,
    "availability_classification.csv"
)

STATUS_FILE = os.path.join(
    FULL_DIR,
    "download_status.csv"
)

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

# HARD timeout for one video
VIDEO_TIMEOUT = 5 * 60       # 5 minutes

SOCKET_TIMEOUT = 30

MAX_RETRIES = 1
FRAGMENT_RETRIES = 1

MAX_HEIGHT = 720


# ============================================================
# FIND LOCAL VIDEO
# ============================================================

def find_local_video(video_id):

    extensions = [
        ".mp4",
        ".mkv",
        ".webm",
        ".mov"
    ]

    for ext in extensions:

        path = os.path.join(
            VIDEO_DIR,
            video_id + ext
        )

        if os.path.isfile(path):
            return path

    return None


# ============================================================
# VALIDATE VIDEO USING FFPROBE
# ============================================================

def validate_video(path):

    if not path:
        return False

    if not os.path.isfile(path):
        return False

    if os.path.getsize(path) < 10000:
        return False

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "default=noprint_wrappers=1",
        path
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return False

        if not result.stdout.strip():
            return False

        return True

    except Exception:
        return False


# ============================================================
# REMOVE PARTIAL DOWNLOADS
# ============================================================

def cleanup_partial(video_id):

    print(
        f"Cleaning partial files for {video_id}..."
    )

    try:

        for filename in os.listdir(VIDEO_DIR):

            if not filename.startswith(video_id):
                continue

            if (
                ".part" in filename
                or ".ytdl" in filename
            ):

                path = os.path.join(
                    VIDEO_DIR,
                    filename
                )

                try:

                    os.remove(path)

                    print(
                        f"Removed: {filename}"
                    )

                except Exception:
                    pass

    except Exception:
        pass


# ============================================================
# SAVE STATUS
# ============================================================

def save_status(status_dict):

    rows = []

    for video_id, info in status_dict.items():

        rows.append({
            "id": video_id,
            "status": info["status"],
            "error": info["error"],
            "filename": info["filename"],
            "timestamp": info["timestamp"]
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        STATUS_FILE,
        index=False
    )


# ============================================================
# DOWNLOAD ONE VIDEO
# ============================================================

def download_video(video_id, url):

    output_template = os.path.join(
        VIDEO_DIR,
        "%(id)s.%(ext)s"
    )

    log_file = os.path.join(
        LOG_DIR,
        f"{video_id}.txt"
    )

    command = [
        "python",
        "-m",
        "yt_dlp",

        # ----------------------------------------------------
        # Cookies
        # ----------------------------------------------------

        "--cookies-from-browser",
        "firefox",

        "--no-playlist",

        # ----------------------------------------------------
        # Format
        # ----------------------------------------------------

        "-f",
        (
            "bestvideo[height<=720][ext=mp4]"
            "+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]/"
            "best[height<=720]"
        ),

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        "-o",
        output_template,

        # ----------------------------------------------------
        # Retry control
        # ----------------------------------------------------

        "--retries",
        str(MAX_RETRIES),

        "--fragment-retries",
        str(FRAGMENT_RETRIES),

        "--socket-timeout",
        str(SOCKET_TIMEOUT),

        "--concurrent-fragments",
        "1",

        # ----------------------------------------------------
        # Don't overwrite existing files
        # ----------------------------------------------------

        "--no-overwrites",

        "--continue",

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url
    ]

    print()
    print("=" * 70)
    print(f"DOWNLOADING: {video_id}")
    print(f"TIMEOUT: {VIDEO_TIMEOUT // 60} minutes")
    print("=" * 70)

    start_time = time.time()

    try:

        with open(
            log_file,
            "w",
            encoding="utf-8"
        ) as log:

            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT
            )

            try:

                return_code = process.wait(
                    timeout=VIDEO_TIMEOUT
                )

            except subprocess.TimeoutExpired:

                print()
                print(
                    f"[TIMEOUT] {video_id}"
                )

                # Kill yt-dlp
                process.kill()

                try:
                    process.wait(timeout=10)
                except Exception:
                    pass

                # Also kill possible child processes
                subprocess.run(
                    [
                        "pkill",
                        "-TERM",
                        "-P",
                        str(process.pid)
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                cleanup_partial(video_id)

                return (
                    "timeout",
                    "",
                    f"Exceeded {VIDEO_TIMEOUT // 60} minute timeout"
                )

        elapsed = time.time() - start_time

        # ----------------------------------------------------
        # Check resulting file
        # ----------------------------------------------------

        local_video = find_local_video(
            video_id
        )

        if (
            return_code == 0
            and local_video
            and validate_video(local_video)
        ):

            print()
            print(
                f"[SUCCESS] {video_id}"
            )

            print(
                f"Time: {elapsed / 60:.2f} minutes"
            )

            return (
                "success",
                local_video,
                ""
            )

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        print()
        print(
            f"[FAILED] {video_id}"
        )

        cleanup_partial(video_id)

        return (
            "failed",
            "",
            f"yt-dlp exit code: {return_code}"
        )

    except KeyboardInterrupt:

        print()
        print(
            f"[INTERRUPTED] {video_id}"
        )

        try:
            process.kill()
        except Exception:
            pass

        raise

    except Exception as e:

        print()
        print(
            f"[ERROR] {video_id}: {e}"
        )

        cleanup_partial(video_id)

        return (
            "error",
            "",
            str(e)
        )


# ============================================================
# LOAD AVAILABILITY FILE
# ============================================================

print("=" * 70)
print("GAVD AVAILABILITY-AWARE DOWNLOADER")
print("=" * 70)

if not os.path.exists(AVAILABILITY_FILE):

    raise FileNotFoundError(
        f"\nAvailability file not found:\n"
        f"{AVAILABILITY_FILE}"
    )

availability = pd.read_csv(
    AVAILABILITY_FILE
)

# ------------------------------------------------------------
# Only genuinely available videos
# ------------------------------------------------------------

available = availability[
    availability["classification"] == "available"
].copy()

available = available.drop_duplicates(
    subset=["id"]
).reset_index(drop=True)

print()
print(
    f"Available videos: {len(available)}"
)


# ============================================================
# LOAD OLD STATUS IF IT EXISTS
# ============================================================

status_dict = {}

if os.path.exists(STATUS_FILE):

    try:

        old_status = pd.read_csv(
            STATUS_FILE
        )

        if "id" in old_status.columns:

            for _, row in old_status.iterrows():

                video_id = str(row["id"])

                status_dict[video_id] = {
                    "status": str(
                        row.get("status", "")
                    ),
                    "error": str(
                        row.get("error", "")
                    ),
                    "filename": str(
                        row.get("filename", "")
                    ),
                    "timestamp": str(
                        row.get("timestamp", "")
                    )
                }

            print(
                f"Loaded existing status: "
                f"{len(status_dict)} entries"
            )

        else:

            print(
                "Existing status file has incompatible "
                "format. Rebuilding status."
            )

    except Exception as e:

        print(
            f"Could not read old status: {e}"
        )

else:

    print(
        "No existing download status found."
    )


# ============================================================
# CHECK EXISTING FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING EXISTING VIDEOS")
print("=" * 70)

already_downloaded = 0

for _, row in available.iterrows():

    video_id = str(row["id"])

    local_video = find_local_video(
        video_id
    )

    if (
        local_video
        and validate_video(local_video)
    ):

        status_dict[video_id] = {
            "status": "success",
            "error": "",
            "filename": local_video,
            "timestamp": datetime.now().isoformat()
        }

        already_downloaded += 1


print()
print(
    f"Valid videos already downloaded: "
    f"{already_downloaded}"
)

print(
    f"Videos still to download: "
    f"{len(available) - already_downloaded}"
)


# ============================================================
# MAIN DOWNLOAD LOOP
# ============================================================

for index, row in available.iterrows():

    video_id = str(row["id"])
    url = str(row["url"])

    print()
    print(
        f"[{index + 1}/{len(available)}] "
        f"{video_id}"
    )

    # --------------------------------------------------------
    # Check existing file
    # --------------------------------------------------------

    local_video = find_local_video(
        video_id
    )

    if (
        local_video
        and validate_video(local_video)
    ):

        print(
            "Already downloaded and valid."
        )

        status_dict[video_id] = {
            "status": "success",
            "error": "",
            "filename": local_video,
            "timestamp": datetime.now().isoformat()
        }

        save_status(status_dict)

        continue

    # --------------------------------------------------------
    # IMPORTANT:
    # Don't retry a video that already failed in THIS
    # download_status unless you manually remove its entry.
    # --------------------------------------------------------

    previous_status = status_dict.get(
        video_id,
        {}
    ).get(
        "status",
        ""
    )

    if previous_status in [
        "timeout",
        "failed",
        "error"
    ]:

        print(
            f"Previously marked as {previous_status}."
        )

        print(
            "Skipping. Delete its status entry "
            "if you want to retry it."
        )

        continue

    # --------------------------------------------------------
    # Clean old partial files
    # --------------------------------------------------------

    cleanup_partial(
        video_id
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    try:

        status, filename, error = download_video(
            video_id,
            url
        )

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("DOWNLOAD INTERRUPTED")
        print("=" * 70)

        print(
            "Saving current progress..."
        )

        save_status(
            status_dict
        )

        print(
            "You can safely run this script again."
        )

        break

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    status_dict[video_id] = {
        "status": status,
        "error": error,
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    }

    save_status(
        status_dict
    )

    # --------------------------------------------------------
    # Small delay between videos
    # --------------------------------------------------------

    time.sleep(1)


# ============================================================
# FINAL SUMMARY
# ============================================================

save_status(
    status_dict
)

print()
print()
print("=" * 70)
print("FINAL DOWNLOAD STATUS")
print("=" * 70)

status_df = pd.DataFrame([
    {
        "id": vid,
        "status": info["status"],
        "error": info["error"],
        "filename": info["filename"]
    }
    for vid, info in status_dict.items()
])

if not status_df.empty:

    print(
        status_df["status"].value_counts()
    )

    successful = (
        status_df["status"] == "success"
    ).sum()

    print()
    print(
        f"Successful: {successful}/{len(available)}"
    )

print()
print(
    f"Status saved to:\n{STATUS_FILE}"
)

print()
print("=" * 70)
print("DOWNLOAD PROCESS FINISHED")
print("=" * 70)