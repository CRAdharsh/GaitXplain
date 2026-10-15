import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "Final Year Project/GAVD-main"

FULL_DIR = os.path.join(BASE_DIR, "full_dataset")

VIDEO_DIR = os.path.join(FULL_DIR, "videos")

AVAILABILITY_FILE = os.path.join(
    FULL_DIR,
    "availability_classification.csv"
)

STATUS_FILE = os.path.join(
    FULL_DIR,
    "download_status.csv"
)

AUDIT_FILE = os.path.join(
    FULL_DIR,
    "video_audit.csv"
)

# ============================================================
# SETTINGS
# ============================================================

# Number of videos checked simultaneously.
# Keep this moderate because FFmpeg can be CPU intensive.
MAX_WORKERS = 4

FFPROBE_TIMEOUT = 60


# ============================================================
# FIND VIDEO
# ============================================================

def find_video(video_id):

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
# AUDIT ONE VIDEO
# ============================================================

def audit_video(row):

    video_id = str(row["id"])

    expected_class = str(
        row.get("gait_pat", "")
    )

    expected_camera = str(
        row.get("cam_view", "")
    )

    path = find_video(video_id)

    result = {
        "id": video_id,
        "gait_pat": expected_class,
        "cam_view": expected_camera,
        "file_path": path if path else "",
        "file_exists": False,
        "file_size_mb": 0,
        "ffprobe_ok": False,
        "codec": "",
        "width": 0,
        "height": 0,
        "fps": 0,
        "frame_count": 0,
        "duration_seconds": 0,
        "pixel_format": "",
        "video_bitrate": 0,
        "audio_codec": "",
        "audio_sample_rate": 0,
        "audio_channels": 0,
        "error": "",
        "audit_status": "missing",
    }

    # --------------------------------------------------------
    # FILE EXISTS?
    # --------------------------------------------------------

    if not path:

        result["error"] = (
            "Video file not found"
        )

        return result

    result["file_exists"] = True

    try:

        size_bytes = os.path.getsize(path)

        result["file_size_mb"] = round(
            size_bytes / (1024 * 1024),
            2
        )

    except Exception as e:

        result["error"] = (
            f"Could not read file size: {e}"
        )

        return result

    # --------------------------------------------------------
    # FFPROBE
    # --------------------------------------------------------

    command = [
        "ffprobe",

        "-v",
        "error",

        # Video stream
        "-select_streams",
        "v:0",

        "-show_entries",
        (
            "stream="
            "codec_name,"
            "width,"
            "height,"
            "r_frame_rate,"
            "avg_frame_rate,"
            "nb_frames,"
            "duration,"
            "pix_fmt,"
            "bit_rate"
        ),

        "-of",
        "default=noprint_wrappers=1",

        path
    ]

    try:

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFPROBE_TIMEOUT
        )

    except subprocess.TimeoutExpired:

        result["error"] = (
            "FFprobe timeout"
        )

        result["audit_status"] = "probe_timeout"

        return result

    except Exception as e:

        result["error"] = str(e)

        result["audit_status"] = "probe_error"

        return result

    # --------------------------------------------------------
    # FFPROBE FAILED
    # --------------------------------------------------------

    if process.returncode != 0:

        result["error"] = (
            process.stderr.strip()
        )

        result["audit_status"] = "corrupt_or_unreadable"

        return result

    # --------------------------------------------------------
    # PARSE FFPROBE
    # --------------------------------------------------------

    values = {}

    for line in process.stdout.splitlines():

        if "=" in line:

            key, value = line.split(
                "=",
                1
            )

            values[key.strip()] = value.strip()

    try:

        result["codec"] = values.get(
            "codec_name",
            ""
        )

        result["width"] = int(
            values.get(
                "width",
                0
            )
        )

        result["height"] = int(
            values.get(
                "height",
                0
            )
        )

        result["pixel_format"] = values.get(
            "pix_fmt",
            ""
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps_string = values.get(
            "avg_frame_rate",
            values.get(
                "r_frame_rate",
                "0/1"
            )
        )

        if "/" in fps_string:

            numerator, denominator = (
                fps_string.split(
                    "/",
                    1
                )
            )

            numerator = float(numerator)
            denominator = float(denominator)

            if denominator != 0:

                result["fps"] = round(
                    numerator / denominator,
                    6
                )

        else:

            result["fps"] = float(
                fps_string
            )

        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        frame_string = values.get(
            "nb_frames",
            ""
        )

        if frame_string:
            result["frame_count"] = int(
                float(frame_string)
            )

        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        duration_string = values.get(
            "duration",
            ""
        )

        if duration_string:

            result["duration_seconds"] = round(
                float(duration_string),
                6
            )

        # ----------------------------------------------------
        # BITRATE
        # ----------------------------------------------------

        bitrate_string = values.get(
            "bit_rate",
            ""
        )

        if bitrate_string:

            result["video_bitrate"] = int(
                float(bitrate_string)
            )

    except Exception as e:

        result["error"] = (
            f"Metadata parsing error: {e}"
        )

        result["audit_status"] = "metadata_error"

        return result

    # ========================================================
    # SECOND FFPROBE
    # Check audio stream separately
    # ========================================================

    audio_command = [
        "ffprobe",

        "-v",
        "error",

        "-select_streams",
        "a:0",

        "-show_entries",
        "stream=codec_name,sample_rate,channels",

        "-of",
        "default=noprint_wrappers=1",

        path
    ]

    try:

        audio_process = subprocess.run(
            audio_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        audio_values = {}

        for line in audio_process.stdout.splitlines():

            if "=" in line:

                key, value = line.split(
                    "=",
                    1
                )

                audio_values[key.strip()] = value.strip()

        result["audio_codec"] = audio_values.get(
            "codec_name",
            ""
        )

        sample_rate = audio_values.get(
            "sample_rate",
            ""
        )

        if sample_rate:

            result["audio_sample_rate"] = int(
                float(sample_rate)
            )

        channels = audio_values.get(
            "channels",
            ""
        )

        if channels:

            result["audio_channels"] = int(
                float(channels)
            )

    except Exception:
        pass

    # ========================================================
    # BASIC VALIDITY CHECKS
    # ========================================================

    problems = []

    if result["file_size_mb"] <= 0:

        problems.append(
            "zero_file_size"
        )

    if result["width"] <= 0:

        problems.append(
            "invalid_width"
        )

    if result["height"] <= 0:

        problems.append(
            "invalid_height"
        )

    if result["fps"] <= 0:

        problems.append(
            "invalid_fps"
        )

    if result["frame_count"] <= 0:

        problems.append(
            "invalid_frame_count"
        )

    if result["duration_seconds"] <= 0:

        problems.append(
            "invalid_duration"
        )

    if result["codec"] == "":

        problems.append(
            "missing_video_codec"
        )

    if problems:

        result["error"] = "; ".join(
            problems
        )

        result["audit_status"] = "invalid"

    else:

        result["audit_status"] = "valid"

    return result


# ============================================================
# LOAD AVAILABILITY LIST
# ============================================================

print("=" * 70)
print("GAVD FULL VIDEO AUDIT")
print("=" * 70)

if not os.path.exists(
    AVAILABILITY_FILE
):

    raise FileNotFoundError(
        AVAILABILITY_FILE
    )

availability = pd.read_csv(
    AVAILABILITY_FILE
)

# ------------------------------------------------------------
# Only videos that should actually exist
# ------------------------------------------------------------

expected = availability[
    availability["classification"] == "available"
].copy()

expected = expected.drop_duplicates(
    subset=["id"]
).reset_index(drop=True)

print()
print(
    f"Expected accessible videos: "
    f"{len(expected)}"
)

# ============================================================
# CHECK DIRECTORY FOR EXTRA FILES
# ============================================================

print()
print(
    "Checking for unexpected video files..."
)

actual_video_files = []

if os.path.isdir(VIDEO_DIR):

    for filename in os.listdir(
        VIDEO_DIR
    ):

        lower = filename.lower()

        if lower.endswith(
            (
                ".mp4",
                ".mkv",
                ".webm",
                ".mov"
            )
        ):

            actual_video_files.append(
                filename
            )

print(
    f"Actual video files found: "
    f"{len(actual_video_files)}"
)


# ============================================================
# AUDIT VIDEOS
# ============================================================

print()
print("=" * 70)
print("RUNNING FFPROBE AUDIT")
print("=" * 70)

results = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            audit_video,
            row
        ): str(row["id"])

        for _, row in expected.iterrows()
    }

    completed = 0

    for future in as_completed(
        futures
    ):

        completed += 1

        video_id = futures[
            future
        ]

        try:

            result = future.result()

        except Exception as e:

            result = {
                "id": video_id,
                "audit_status": "exception",
                "error": str(e)
            }

        results.append(
            result
        )

        status = result.get(
            "audit_status",
            "unknown"
        )

        print(
            f"[{completed}/{len(expected)}] "
            f"{video_id} -> {status}"
        )


# ============================================================
# CREATE AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame(
    results
)

audit = audit.sort_values(
    "id"
).reset_index(drop=True)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(
    audit["audit_status"]
    .value_counts(
        dropna=False
    )
)

valid_count = (
    audit["audit_status"] == "valid"
).sum()

print()
print(
    f"Valid videos: "
    f"{valid_count}/{len(expected)}"
)

# ============================================================
# MISSING
# ============================================================

missing = audit[
    audit["audit_status"] == "missing"
]

if len(missing) > 0:

    print()
    print("=" * 70)
    print("MISSING VIDEOS")
    print("=" * 70)

    print(
        missing[
            ["id", "gait_pat", "cam_view"]
        ].to_string(
            index=False
        )
    )


# ============================================================
# INVALID / CORRUPT
# ============================================================

invalid = audit[
    audit["audit_status"].isin(
        [
            "corrupt_or_unreadable",
            "invalid",
            "metadata_error",
            "probe_error",
            "probe_timeout",
            "exception"
        ]
    )
]

if len(invalid) > 0:

    print()
    print("=" * 70)
    print("INVALID / PROBLEMATIC VIDEOS")
    print("=" * 70)

    print(
        invalid[
            [
                "id",
                "gait_pat",
                "cam_view",
                "file_size_mb",
                "codec",
                "fps",
                "frame_count",
                "duration_seconds",
                "error"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FPS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("FPS DISTRIBUTION")
print("=" * 70)

valid = audit[
    audit["audit_status"] == "valid"
]

if len(valid) > 0:

    fps_distribution = (
        valid["fps"]
        .round(3)
        .value_counts()
        .sort_index()
    )

    print(
        fps_distribution.to_string()
    )


# ============================================================
# RESOLUTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("RESOLUTION DISTRIBUTION")
print("=" * 70)

if len(valid) > 0:

    resolution = (
        valid["width"].astype(str)
        + "x"
        + valid["height"].astype(str)
    )

    print(
        resolution.value_counts()
        .to_string()
    )


# ============================================================
# CODEC DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("VIDEO CODEC DISTRIBUTION")
print("=" * 70)

if len(valid) > 0:

    print(
        valid["codec"]
        .value_counts()
        .to_string()
    )


# ============================================================
# FRAME STATISTICS
# ============================================================

print()
print("=" * 70)
print("FRAME COUNT STATISTICS")
print("=" * 70)

if len(valid) > 0:

    print(
        valid["frame_count"]
        .describe()
        .to_string()
    )


# ============================================================
# DURATION STATISTICS
# ============================================================

print()
print("=" * 70)
print("DURATION STATISTICS")
print("=" * 70)

if len(valid) > 0:

    print(
        valid["duration_seconds"]
        .describe()
        .to_string()
    )


# ============================================================
# EXTRA FILES
# ============================================================

expected_ids = set(
    expected["id"].astype(str)
)

extra_files = []

for filename in actual_video_files:

    video_id = os.path.splitext(
        filename
    )[0]

    if video_id not in expected_ids:

        extra_files.append(
            filename
        )

print()
print("=" * 70)
print("EXTRA VIDEO FILES")
print("=" * 70)

if extra_files:

    for filename in extra_files:

        print(filename)

else:

    print(
        "No unexpected video files."
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    f"Audit CSV:\n{AUDIT_FILE}"
)

print()
print(
    f"Audit completed at: "
    f"{datetime.now()}"
)

print("=" * 70)