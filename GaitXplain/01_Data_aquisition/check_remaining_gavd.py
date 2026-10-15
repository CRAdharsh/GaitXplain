import pandas as pd
import subprocess
import os
import re
import time


# ============================================================
# PATHS
# ============================================================

BASE = "/Final Year Project"

GAVD_DIR = os.path.join(
    BASE,
    "GAVD-main"
)

DATASET_DIR = os.path.join(
    GAVD_DIR,
    "full_dataset"
)

STATUS_FILE = os.path.join(
    DATASET_DIR,
    "download_status.csv"
)

OUTPUT_FILE = os.path.join(
    DATASET_DIR,
    "remaining_video_check.csv"
)


# ============================================================
# LOAD STATUS
# ============================================================

if not os.path.exists(STATUS_FILE):

    print(
        "ERROR: download_status.csv not found."
    )

    raise SystemExit


status = pd.read_csv(
    STATUS_FILE,
    dtype=str
).fillna("")


# ============================================================
# FIND REMAINING VIDEOS
# ============================================================

remaining = status[
    status["status"] != "success"
].copy()


print("=" * 80)
print("GAVD REMAINING VIDEO AVAILABILITY CHECK")
print("=" * 80)

print()

print(
    "Total entries:",
    len(status)
)

print(
    "Already successful:",
    (
        status["status"]
        == "success"
    ).sum()
)

print(
    "Need checking:",
    len(remaining)
)

print()


# ============================================================
# CHECK EACH VIDEO
# ============================================================

results = []


for i, row in remaining.iterrows():

    video_id = row["video_id"]
    url = row["url"]

    print("=" * 80)

    print(
        f"[{len(results) + 1}/{len(remaining)}]"
    )

    print(
        "Video:",
        video_id
    )

    print(
        "Class:",
        row.get(
            "gait_class",
            ""
        )
    )

    print(
        "URL:",
        url
    )

    print()


    command = [

        "python",
        "-m",
        "yt_dlp",

        "--cookies-from-browser",
        "firefox",

        "--no-playlist",

        "--skip-download",

        "--print",
        "%(id)s",

        "--print",
        "%(title)s",

        "--print",
        "%(duration)s",

        "--print",
        "%(availability)s",

        url
    ]


    try:

        result = subprocess.run(

            command,

            capture_output=True,

            text=True,

            timeout=60
        )


        combined = (
            result.stdout
            + "\n"
            + result.stderr
        )


        if result.returncode == 0:

            print(
                "AVAILABLE"
            )

            status_result = "available"

        else:

            lower = combined.lower()

            if (
                "this video is unavailable"
                in lower
            ):

                status_result = "unavailable"

            elif (
                "private video"
                in lower
            ):

                status_result = "private"

            elif (
                "video has been removed"
                in lower
            ):

                status_result = "removed"

            elif (
                "members-only"
                in lower
            ):

                status_result = "members_only"

            elif (
                "sign in"
                in lower
                and "confirm your age"
                in lower
            ):

                status_result = "age_restricted"

            elif (
                "login required"
                in lower
            ):

                status_result = "login_required"

            else:

                status_result = "other_error"


            print(
                "RESULT:",
                status_result
            )


        results.append({

            "video_id":
                video_id,

            "url":
                url,

            "gait_class":
                row.get(
                    "gait_class",
                    ""
                ),

            "camera_view":
                row.get(
                    "camera_view",
                    ""
                ),

            "result":
                status_result,

            "return_code":
                result.returncode,

            "output":
                combined[-3000:]
        })


    except subprocess.TimeoutExpired:

        print(
            "CHECK TIMEOUT"
        )

        results.append({

            "video_id":
                video_id,

            "url":
                url,

            "gait_class":
                row.get(
                    "gait_class",
                    ""
                ),

            "camera_view":
                row.get(
                    "camera_view",
                    ""
                ),

            "result":
                "check_timeout",

            "return_code":
                "",

            "output":
                "yt-dlp availability check exceeded 60 seconds"
        })


    except Exception as e:

        print(
            "ERROR:",
            e
        )

        results.append({

            "video_id":
                video_id,

            "url":
                url,

            "gait_class":
                row.get(
                    "gait_class",
                    ""
                ),

            "camera_view":
                row.get(
                    "camera_view",
                    ""
                ),

            "result":
                "python_error",

            "return_code":
                "",

            "output":
                str(e)
        })


    # Save after every video
    pd.DataFrame(
        results
    ).to_csv(
        OUTPUT_FILE,
        index=False
    )

    time.sleep(0.5)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("CHECK FINISHED")
print("=" * 80)

print()

print(
    results_df[
        "result"
    ].value_counts()
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)