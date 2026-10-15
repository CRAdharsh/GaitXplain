import pandas as pd
import subprocess
import os
import json
import time

# ============================================================
# PATHS
# ============================================================

BASE = "/Final Year Project"

GAVD_DIR = os.path.join(
    BASE,
    "GAVD-main"
)

# Official annotation CSVs
ANNOTATION_FILES = [
    os.path.join(GAVD_DIR, "data", "GAVD_Clinical_Annotations_1.csv"),
    os.path.join(GAVD_DIR, "data", "GAVD_Clinical_Annotations_2.csv"),
    os.path.join(GAVD_DIR, "data", "GAVD_Clinical_Annotations_3.csv"),
    os.path.join(GAVD_DIR, "data", "GAVD_Clinical_Annotations_4.csv"),
    os.path.join(GAVD_DIR, "data", "GAVD_Clinical_Annotations_5.csv"),
]

# Full dataset download directory
DATASET_DIR = os.path.join(
    GAVD_DIR,
    "full_dataset"
)

VIDEO_DIR = os.path.join(
    DATASET_DIR,
    "videos"
)

LOG_DIR = os.path.join(
    DATASET_DIR,
    "download_logs"
)

STATUS_FILE = os.path.join(
    DATASET_DIR,
    "download_status.csv"
)

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def verify_video(video_file):

    """
    Use FFmpeg to verify that the downloaded video
    can actually be decoded.

    Returns:
        True  -> valid video
        False -> invalid/corrupt video
    """

    if not os.path.exists(video_file):
        return False

    if os.path.getsize(video_file) == 0:
        return False

    command = [
        "ffmpeg",

        "-v",
        "error",

        "-i",
        video_file,

        # Decode to null output
        "-f",
        "null",

        "-"
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            return True

        return False

    except Exception:

        return False


def get_video_information(video_file):

    """
    Get basic information from FFprobe.
    """

    command = [
        "ffprobe",

        "-v",
        "quiet",

        "-print_format",
        "json",

        "-show_streams",

        "-show_format",

        video_file
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return {}

        return json.loads(result.stdout)

    except Exception:

        return {}


def load_status():

    """
    Load previous download status if it exists.
    """

    if os.path.exists(STATUS_FILE):

        try:

            return pd.read_csv(
                STATUS_FILE,
                dtype=str
            )

        except Exception:

            print("WARNING: Could not read existing status file.")
            print("Starting with a fresh status table.")

    return pd.DataFrame()


def save_status(status_df):

    """
    Save current download status.
    """

    status_df.to_csv(
        STATUS_FILE,
        index=False
    )


# ============================================================
# READ ALL GAVD ANNOTATIONS
# ============================================================

print("=" * 80)
print("GAVD FULL DATASET DOWNLOADER")
print("=" * 80)

print()
print("Reading annotation files...")
print()

all_annotations = []

for annotation_file in ANNOTATION_FILES:

    print("Reading:")
    print(annotation_file)

    if not os.path.exists(annotation_file):

        print("ERROR: File does not exist.")
        print()

        continue

    try:

        df = pd.read_csv(
            annotation_file
        )

        print(
            "Rows:",
            len(df)
        )

        print(
            "Columns:",
            list(df.columns)
        )

        all_annotations.append(df)

    except Exception as e:

        print(
            "ERROR reading file:",
            e
        )

    print()


# ============================================================
# COMBINE ANNOTATIONS
# ============================================================

if len(all_annotations) == 0:

    print("ERROR: No annotation files could be read.")
    exit()


annotations = pd.concat(
    all_annotations,
    ignore_index=True
)

print("=" * 80)
print("ANNOTATION DATASET")
print("=" * 80)

print(
    "Total annotation rows:",
    len(annotations)
)

print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id",
    "url",
    "gait_pat",
    "cam_view"
]

missing_columns = [
    col
    for col in required_columns
    if col not in annotations.columns
]

if missing_columns:

    print(
        "ERROR: Missing required columns:"
    )

    for col in missing_columns:
        print(
            "  -",
            col
        )

    exit()


# ============================================================
# CREATE UNIQUE VIDEO TABLE
# ============================================================

videos = (
    annotations[
        [
            "id",
            "url",
            "gait_pat",
            "cam_view"
        ]
    ]
    .dropna(
        subset=["id", "url"]
    )
    .drop_duplicates(
        subset=["id"]
    )
    .reset_index(
        drop=True
    )
)

print(
    "Unique videos:",
    len(videos)
)

print()


# ============================================================
# LOAD PREVIOUS STATUS
# ============================================================

old_status = load_status()


# Convert previous status into dictionary
status_dict = {}

if len(old_status) > 0:

    for _, row in old_status.iterrows():

        video_id = str(
            row.get(
                "video_id",
                ""
            )
        )

        if video_id:

            status_dict[video_id] = row.to_dict()


# ============================================================
# DOWNLOAD LOOP
# ============================================================

print("=" * 80)
print("STARTING DOWNLOAD")
print("=" * 80)

print()


for i, row in videos.iterrows():

    video_id = str(
        row["id"]
    )

    url = str(
        row["url"]
    )

    gait_class = str(
        row["gait_pat"]
    )

    camera_view = str(
        row["cam_view"]
    )

    output_file = os.path.join(
        VIDEO_DIR,
        video_id + ".mp4"
    )

    log_file = os.path.join(
        LOG_DIR,
        video_id + "_download.log"
    )

    print("=" * 80)

    print(
        f"[{i + 1}/{len(videos)}]"
    )

    print(
        "Video ID:",
        video_id
    )

    print(
        "Class:",
        gait_class
    )

    print(
        "View:",
        camera_view
    )

    print(
        "URL:",
        url
    )

    # --------------------------------------------------------
    # Check existing verified download
    # --------------------------------------------------------

    previous = status_dict.get(
        video_id,
        {}
    )

    previous_status = previous.get(
        "status",
        ""
    )

    if (
        previous_status == "success"
        and os.path.exists(output_file)
    ):

        print(
            "Previously verified successfully."
        )

        print(
            "Skipping."
        )

        continue


    # --------------------------------------------------------
    # If file exists but status is unknown,
    # verify it before downloading again.
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print(
            "Existing file found."
        )

        print(
            "Verifying with FFmpeg..."
        )

        if verify_video(output_file):

            print(
                "Existing file is valid."
            )

            status_dict[video_id] = {

                "video_id": video_id,

                "url": url,

                "gait_class": gait_class,

                "camera_view": camera_view,

                "status": "success",

                "filename": output_file,

                "file_size_mb":
                    round(
                        os.path.getsize(
                            output_file
                        ) / (1024 * 1024),
                        2
                    ),

                "error": ""
            }

            # Save immediately
            status_df = pd.DataFrame(
                list(
                    status_dict.values()
                )
            )

            save_status(
                status_df
            )

            print(
                "Skipping download."
            )

            continue

        else:

            print(
                "Existing file is invalid."
            )

            print(
                "Deleting it."
            )

            try:

                os.remove(
                    output_file
                )

            except Exception:

                pass


    # --------------------------------------------------------
    # yt-dlp command
    # --------------------------------------------------------

    command = [

        "python",

        "-m",

        "yt_dlp",

        # Firefox authentication
        "--cookies-from-browser",

        "firefox",

        # Best MP4 video + M4A audio
        "-f",

        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",

        # Merge into MP4
        "--merge-output-format",

        "mp4",

        # Don't download playlists
        "--no-playlist",

        # Don't stop entire process on failure
        "--no-abort-on-error",

        # Retry network failures
        "--retries",

        "5",

        # Retry fragments
        "--fragment-retries",

        "5",

        # Output filename
        "-o",

        output_file,

        url
    ]


    # --------------------------------------------------------
    # Run yt-dlp
    # --------------------------------------------------------

    print()
    print(
        "Starting yt-dlp..."
    )
    print()

    start_time = time.time()

    try:

        result = subprocess.run(

            command,

            capture_output=True,

            text=True
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Save complete log
        # ----------------------------------------------------

        with open(
            log_file,
            "w"
        ) as f:

            f.write(
                "COMMAND:\n"
            )

            f.write(
                " ".join(command)
            )

            f.write(
                "\n\nSTDOUT:\n"
            )

            f.write(
                result.stdout
            )

            f.write(
                "\n\nSTDERR:\n"
            )

            f.write(
                result.stderr
            )


        # ----------------------------------------------------
        # Verify result
        # ----------------------------------------------------

        print()

        print(
            "Download command finished."
        )

        print(
            "Elapsed:",
            round(
                elapsed,
                1
            ),
            "seconds"
        )

        if os.path.exists(
            output_file
        ):

            print(
                "Checking downloaded video with FFmpeg..."
            )

            valid = verify_video(
                output_file
            )

        else:

            valid = False


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if valid:

            file_size_mb = round(
                os.path.getsize(
                    output_file
                ) / (1024 * 1024),
                2
            )

            print()
            print(
                "DOWNLOAD SUCCESS"
            )

            print(
                "File size:",
                file_size_mb,
                "MB"
            )


            # Get video information
            info = get_video_information(
                output_file
            )

            fps = ""
            frame_count = ""

            try:

                video_stream = next(
                    stream
                    for stream in info.get(
                        "streams",
                        []
                    )
                    if stream.get(
                        "codec_type"
                    ) == "video"
                )

                fps = video_stream.get(
                    "r_frame_rate",
                    ""
                )

                frame_count = video_stream.get(
                    "nb_frames",
                    ""
                )

            except Exception:

                pass


            status_dict[video_id] = {

                "video_id": video_id,

                "url": url,

                "gait_class": gait_class,

                "camera_view": camera_view,

                "status": "success",

                "filename": output_file,

                "file_size_mb": file_size_mb,

                "fps": fps,

                "frame_count": frame_count,

                "error": ""
            }


        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        else:

            print()
            print(
                "DOWNLOAD FAILED"
            )

            print(
                "See log:"
            )

            print(
                log_file
            )


            error_text = (
                result.stderr
                .strip()
            )

            status_dict[video_id] = {

                "video_id": video_id,

                "url": url,

                "gait_class": gait_class,

                "camera_view": camera_view,

                "status": "failed",

                "filename": "",

                "file_size_mb": "",

                "fps": "",

                "frame_count": "",

                "error": error_text[-2000:]
            }


    except Exception as e:

        print()
        print(
            "PYTHON ERROR:",
            e
        )

        status_dict[video_id] = {

            "video_id": video_id,

            "url": url,

            "gait_class": gait_class,

            "camera_view": camera_view,

            "status": "error",

            "filename": "",

            "file_size_mb": "",

            "fps": "",

            "frame_count": "",

            "error": str(e)
        }


    # --------------------------------------------------------
    # SAVE STATUS AFTER EVERY VIDEO
    # --------------------------------------------------------

    status_df = pd.DataFrame(
        list(
            status_dict.values()
        )
    )

    save_status(
        status_df
    )

    print()
    print(
        "Status saved."
    )

    # Small delay between videos
    time.sleep(1)


# ============================================================
# FINAL SUMMARY
# ============================================================

status_df = pd.DataFrame(
    list(
        status_dict.values()
    )
)

save_status(
    status_df
)

print()
print("=" * 80)
print("DOWNLOAD PROCESS FINISHED")
print("=" * 80)

print()

if len(status_df) > 0:

    print(
        "Total unique videos:",
        len(videos)
    )

    print()

    print(
        "Successful:",
        (
            status_df["status"]
            == "success"
        ).sum()
    )

    print(
        "Failed:",
        (
            status_df["status"]
            == "failed"
        ).sum()
    )

    print(
        "Errors:",
        (
            status_df["status"]
            == "error"
        ).sum()
    )

    print()

    print(
        "Status file:"
    )

    print(
        STATUS_FILE
    )

    print()

    print(
        "Videos:"
    )

    print(
        VIDEO_DIR
    )

    print()

    print(
        "Logs:"
    )

    print(
        LOG_DIR
    )

print()
print("=" * 80)