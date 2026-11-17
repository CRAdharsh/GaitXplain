import os
import cv2
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE = "/media/legion/Data/Final Year Project/New preprocessing/GAVD-main"

STAGE2 = os.path.join(
    BASE,
    "stage2_test"
)

VIDEO_DIR = os.path.join(
    STAGE2,
    "videos"
)

ANNOTATION_FILE = os.path.join(
    BASE,
    "analysis",
    "GAVD_combined_annotations.csv"
)

VIDEO_INFO_FILE = os.path.join(
    STAGE2,
    "video_info.csv"
)

SEQUENCE_ALIGNMENT_FILE = os.path.join(
    STAGE2,
    "annotation_sequence_alignment.csv"
)

OUTPUT_FILE = os.path.join(
    STAGE2,
    "2sec_video_check.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SECONDS = 2.0

# 50% overlap
OVERLAP = 0.50

STRIDE_SECONDS = (
    WINDOW_SECONDS *
    (1.0 - OVERLAP)
)


# ============================================================
# LOAD INFORMATION
# ============================================================

print("=" * 75)
print("GAVD 2-SECOND WINDOW VIDEO VALIDATION")
print("=" * 75)


print("\nLoading files...")

annotations = pd.read_csv(
    ANNOTATION_FILE
)

video_info = pd.read_csv(
    VIDEO_INFO_FILE
)

sequence_info = pd.read_csv(
    SEQUENCE_ALIGNMENT_FILE
)


print(
    f"Annotation rows: {len(annotations):,}"
)

print(
    f"Video information rows: {len(video_info)}"
)

print(
    f"Sequence information rows: {len(sequence_info)}"
)


# ============================================================
# FIND VIDEO COLUMN
# ============================================================

print("\nAnnotation columns:")

print(
    list(annotations.columns)
)


# ============================================================
# HELPER: FIND COLUMN
# ============================================================

def find_column(
    dataframe,
    possible_names
):

    for name in possible_names:

        if name in dataframe.columns:
            return name

    return None


video_id_col = find_column(
    annotations,
    [
        "id",
        "video_id",
        "video"
    ]
)

sequence_col = find_column(
    annotations,
    [
        "seq",
        "sequence_id",
        "sequence"
    ]
)

frame_col = find_column(
    annotations,
    [
        "frame_num",
        "frame_number",
        "frame"
    ]
)


if video_id_col is None:

    raise ValueError(
        "Could not find video ID column "
        "in annotation CSV."
    )


if sequence_col is None:

    raise ValueError(
        "Could not find sequence ID column "
        "in annotation CSV."
    )


if frame_col is None:

    raise ValueError(
        "Could not find frame-number column "
        "in annotation CSV."
    )


print("\nDetected columns:")

print(
    f"Video ID : {video_id_col}"
)

print(
    f"Sequence : {sequence_col}"
)

print(
    f"Frame    : {frame_col}"
)


# ============================================================
# VIDEO INFO COLUMN DETECTION
# ============================================================

video_info_id_col = find_column(
    video_info,
    [
        "id",
        "video_id",
        "video"
    ]
)

video_info_fps_col = find_column(
    video_info,
    [
        "fps"
    ]
)

video_info_frame_col = find_column(
    video_info,
    [
        "frame_count",
        "frames"
    ]
)


if video_info_id_col is None:
    raise ValueError(
        "Could not find video ID in video_info.csv"
    )


# ============================================================
# SELECT DOWNLOADED VIDEOS
# ============================================================

video_files = []

for filename in os.listdir(VIDEO_DIR):

    if filename.lower().endswith(
        (
            ".mp4",
            ".mkv",
            ".webm",
            ".mov",
            ".avi"
        )
    ):

        video_id = os.path.splitext(
            filename
        )[0]

        video_files.append(
            (
                video_id,
                os.path.join(
                    VIDEO_DIR,
                    filename
                )
            )
        )


video_files.sort()


print("\n" + "=" * 75)
print("DOWNLOADED VIDEOS")
print("=" * 75)

print(
    f"Videos found: {len(video_files)}"
)

for video_id, path in video_files:

    size_mb = (
        os.path.getsize(path)
        / (1024 * 1024)
    )

    print(
        f"{video_id:20s} "
        f"{size_mb:10.1f} MB"
    )


# ============================================================
# PREPARE ANNOTATIONS
# ============================================================

annotations[frame_col] = pd.to_numeric(
    annotations[frame_col],
    errors="coerce"
)

annotations = annotations.dropna(
    subset=[
        video_id_col,
        sequence_col,
        frame_col
    ]
)

annotations[frame_col] = (
    annotations[frame_col]
    .astype(int)
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_id, video_path in video_files:

    print("\n" + "=" * 75)

    print(
        f"VIDEO: {video_id}"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        results.append({
            "video_id": video_id,
            "status": "VIDEO_OPEN_FAILED"
        })

        continue


    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    actual_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    duration = (
        actual_frame_count /
        actual_fps
        if actual_fps > 0
        else 0
    )


    print(
        f"FPS          : {actual_fps:.6f}"
    )

    print(
        f"Frame count  : {actual_frame_count}"
    )

    print(
        f"Duration     : {duration:.2f} sec"
    )

    print(
        f"Resolution   : {width} x {height}"
    )


    # --------------------------------------------------------
    # Get sequences for this video
    # --------------------------------------------------------

    video_annotations = annotations[
        annotations[video_id_col].astype(str)
        == str(video_id)
    ]


    if len(video_annotations) == 0:

        print(
            "WARNING: No annotations found."
        )

        cap.release()

        continue


    sequence_ids = (
        video_annotations[
            sequence_col
        ]
        .dropna()
        .unique()
    )


    print(
        f"Annotated sequences: "
        f"{len(sequence_ids)}"
    )


    # --------------------------------------------------------
    # Process sequences
    # --------------------------------------------------------

    for sequence_id in sequence_ids:

        seq = video_annotations[
            video_annotations[
                sequence_col
            ]
            == sequence_id
        ]


        frame_numbers = np.sort(
            seq[frame_col]
            .unique()
        )


        if len(frame_numbers) == 0:
            continue


        start_frame = int(
            frame_numbers.min()
        )

        end_frame = int(
            frame_numbers.max()
        )


        annotated_count = len(
            frame_numbers
        )


        # ----------------------------------------------------
        # Check continuity
        # ----------------------------------------------------

        expected_frames = np.arange(
            start_frame,
            end_frame + 1
        )

        missing = np.setdiff1d(
            expected_frames,
            frame_numbers
        )


        # ----------------------------------------------------
        # Determine indexing convention
        # ----------------------------------------------------

        # OpenCV frames are 0-based:
        #
        # 0 ... frame_count-1
        #
        # GAVD appears to contain 1-based frame numbers
        # for some sequences.
        #
        # We therefore test both possibilities.

        zero_based_valid = (
            start_frame >= 0
            and end_frame <
            actual_frame_count
        )

        one_based_valid = (
            start_frame >= 1
            and end_frame <=
            actual_frame_count
        )


        # ----------------------------------------------------
        # Sequence duration
        # ----------------------------------------------------

        sequence_duration = (
            annotated_count /
            actual_fps
            if actual_fps > 0
            else 0
        )


        # ----------------------------------------------------
        # Number of possible 2-sec windows
        # ----------------------------------------------------

        if sequence_duration >= WINDOW_SECONDS:

            possible_windows = (
                int(
                    np.floor(
                        (
                            sequence_duration
                            - WINDOW_SECONDS
                        )
                        /
                        STRIDE_SECONDS
                    )
                )
                + 1
            )

        else:

            possible_windows = 0


        # ----------------------------------------------------
        # Test actual frame ranges
        # ----------------------------------------------------

        valid_windows = 0
        invalid_windows = 0

        window_records = []


        if possible_windows > 0:

            for window_index in range(
                possible_windows
            ):

                start_time = (
                    window_index *
                    STRIDE_SECONDS
                )

                end_time = (
                    start_time +
                    WINDOW_SECONDS
                )


                # Convert time to frame positions
                #
                # We use the annotated sequence's
                # first frame as the temporal origin.

                start_offset = int(
                    round(
                        start_time *
                        actual_fps
                    )
                )

                end_offset = int(
                    round(
                        end_time *
                        actual_fps
                    )
                )


                start_idx = (
                    start_offset
                )

                end_idx = (
                    end_offset - 1
                )


                if (
                    start_idx <
                    annotated_count
                    and
                    end_idx <
                    annotated_count
                ):

                    valid_windows += 1

                    window_status = (
                        "VALID"
                    )

                else:

                    invalid_windows += 1

                    window_status = (
                        "INVALID"
                    )


                window_records.append(
                    {
                        "window_index":
                            window_index,

                        "start_time":
                            start_time,

                        "end_time":
                            end_time,

                        "status":
                            window_status
                    }
                )


        # ----------------------------------------------------
        # Print sequence result
        # ----------------------------------------------------

        print(
            f"\nSequence: {sequence_id}"
        )

        print(
            f"  Frames: "
            f"{start_frame} - "
            f"{end_frame}"
        )

        print(
            f"  Annotated frames: "
            f"{annotated_count}"
        )

        print(
            f"  Duration: "
            f"{sequence_duration:.2f} sec"
        )

        print(
            f"  Missing frames: "
            f"{len(missing)}"
        )

        print(
            f"  0-based valid: "
            f"{zero_based_valid}"
        )

        print(
            f"  1-based valid: "
            f"{one_based_valid}"
        )

        print(
            f"  Possible 2-sec windows: "
            f"{possible_windows}"
        )

        print(
            f"  Valid windows: "
            f"{valid_windows}"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "video_id":
                video_id,

            "sequence_id":
                sequence_id,

            "gait_pat":
                seq["gait_pat"].iloc[0]
                if "gait_pat" in seq.columns
                else "",

            "cam_view":
                seq["cam_view"].iloc[0]
                if "cam_view" in seq.columns
                else "",

            "fps":
                actual_fps,

            "video_frame_count":
                actual_frame_count,

            "sequence_start_frame":
                start_frame,

            "sequence_end_frame":
                end_frame,

            "annotated_frame_count":
                annotated_count,

            "missing_frame_count":
                len(missing),

            "sequence_duration":
                sequence_duration,

            "possible_2sec_windows":
                possible_windows,

            "valid_2sec_windows":
                valid_windows,

            "invalid_2sec_windows":
                invalid_windows,

            "zero_based_valid":
                zero_based_valid,

            "one_based_valid":
                one_based_valid,

            "status":
                "OK"
        })


    cap.release()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FINAL VALIDATION SUMMARY")
print("=" * 75)


if len(results_df) > 0:

    print(
        f"\nSequences checked: "
        f"{len(results_df)}"
    )

    print(
        f"Total possible 2-sec windows: "
        f"{results_df['possible_2sec_windows'].sum()}"
    )

    print(
        f"Total valid 2-sec windows: "
        f"{results_df['valid_2sec_windows'].sum()}"
    )

    print(
        f"Total invalid 2-sec windows: "
        f"{results_df['invalid_2sec_windows'].sum()}"
    )

    print(
        f"Sequences with missing frames: "
        f"{(results_df['missing_frame_count'] > 0).sum()}"
    )

    print(
        f"Sequences valid with 0-based indexing: "
        f"{results_df['zero_based_valid'].sum()}"
    )

    print(
        f"Sequences valid with 1-based indexing: "
        f"{results_df['one_based_valid'].sum()}"
    )


print(
    f"\nDetailed results saved to:"
)

print(
    OUTPUT_FILE
)

print("\nDone.")