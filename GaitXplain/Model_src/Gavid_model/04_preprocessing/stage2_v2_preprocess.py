import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/Final Year Project/GAVD-main"
)

FULL_DATASET_DIR = BASE_DIR / "full_dataset"

VIDEO_DIR = FULL_DATASET_DIR / "videos"

AVAILABILITY_CSV = (
    FULL_DATASET_DIR / "availability_classification.csv"
)

SEQUENCE_SUMMARY = (
    BASE_DIR / "analysis" / "sequence_summary.csv"
)

OUTPUT_DIR = (
    FULL_DATASET_DIR / "v2_2sec"
)

OUTPUT_SEQUENCE_DIR = (
    OUTPUT_DIR / "sequences"
)

YOLO_SEG_MODEL = "yolo11n-seg.pt"
YOLO_POSE_MODEL = "yolo11n-pose.pt"

# ------------------------------------------------------------
# Temporal representation
# ------------------------------------------------------------

WINDOW_SECONDS = 2.0
OVERLAP = 0.50
TOKENS_PER_WINDOW = 32

# ------------------------------------------------------------
# Spatial representation
# ------------------------------------------------------------

TARGET_H = 256
TARGET_W = 128

# Exact V1-style segmentation parameters
SEG_CONF = 0.25
MORPH_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 500
ROI_PADDING = 10

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames,duration,width,height",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    if not data.get("streams"):
        raise RuntimeError(
            f"No video stream found: {video_path}"
        )

    stream = data["streams"][0]

    fps_string = stream.get(
        "r_frame_rate",
        "0/1"
    )

    numerator, denominator = fps_string.split("/")
    fps = float(numerator) / float(denominator)

    duration = float(
        stream.get("duration") or 0
    )

    nb_frames = stream.get("nb_frames")

    if nb_frames is not None:
        nb_frames = int(nb_frames)
    else:
        nb_frames = int(round(duration * fps))

    width = int(stream["width"])
    height = int(stream["height"])

    return {
        "fps": fps,
        "duration": duration,
        "frames": nb_frames,
        "width": width,
        "height": height,
    }


# ============================================================
# FFMPEG STREAMING DECODER
# ============================================================

def stream_ffmpeg_frames(
    video_path,
    frame_indices,
    width,
    height,
    frame_callback,
):
    """
    Stream raw BGR frames from FFmpeg.

    No temporary raw-video file is created, so even very large
    GAVD videos do not consume hundreds of GB of temporary space.
    """

    frame_indices = sorted(set(frame_indices))

    if not frame_indices:
        return 0, 0

    wanted = set(frame_indices)

    print(
        f"  FFmpeg streaming {len(wanted)} required frames..."
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )

    frame_size = width * height * 3

    current_index = 0
    decoded_total = 0
    required_decoded = 0

    try:
        while True:

            data = process.stdout.read(frame_size)

            if len(data) != frame_size:
                break

            decoded_total += 1

            if current_index in wanted:

                frame = np.frombuffer(
                    data,
                    dtype=np.uint8
                ).reshape(
                    height,
                    width,
                    3
                ).copy()

                frame_callback(
                    current_index,
                    frame
                )

                required_decoded += 1

            current_index += 1

        stderr_data = process.stderr.read()
        return_code = process.wait()

        if return_code != 0:

            error_text = stderr_data.decode(
                "utf-8",
                errors="replace"
            ).strip()

            raise RuntimeError(
                "FFmpeg failed with return code "
                f"{return_code}:\n{error_text}"
            )

        print(
            f"  Decoded {required_decoded}/"
            f"{len(wanted)} required frames "
            f"(scanned {decoded_total} video frames)"
        )

        return decoded_total, required_decoded

    finally:

        if process.poll() is None:
            process.kill()
            process.wait()

        if process.stdout:
            process.stdout.close()

        if process.stderr:
            process.stderr.close()


# ============================================================
# LOAD SEQUENCE SUMMARY
# ============================================================

def load_sequence_summary():

    rows = []

    with open(
        SEQUENCE_SUMMARY,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "seq",
            "video_id",
            "gait_class",
            "camera_view",
            "first_frame",
            "last_frame",
            "number_of_frames",
            "frame_span",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "sequence_summary.csv is missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:

            rows.append({
                "sequence_id": row["seq"],
                "video_id": row["video_id"],
                "gait_pat": row["gait_class"],
                "cam_view": row["camera_view"],
                "start_frame": int(row["first_frame"]),
                "end_frame": int(row["last_frame"]),
                "number_of_frames": int(
                    row["number_of_frames"]
                ),
                "frame_span": int(
                    row["frame_span"]
                ),
            })

    return rows


# ============================================================
# LOAD ACCESSIBLE VIDEO IDS
# ============================================================

def load_available_video_ids():

    if not AVAILABILITY_CSV.exists():
        raise FileNotFoundError(
            AVAILABILITY_CSV
        )

    df_rows = []

    with open(
        AVAILABILITY_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if (
                row.get("classification", "")
                == "available"
            ):
                df_rows.append(
                    row["id"]
                )

    return set(df_rows)


# ============================================================
# V1-STYLE SEGMENTATION / EDGE EXTRACTION
# ============================================================

def extract_person_segmentation(
    frame,
    seg_model,
):
    """
    V1-style spatial preprocessing:

        YOLO segmentation
        -> largest person by bbox area
        -> polygon mask
        -> morphological closing
        -> largest connected component
        -> minimum component area
        -> 10 px padded ROI
        -> aspect-preserving resize
        -> zero padding
        -> grayscale
        -> Canny(50,150)

    Returns:
        (edges, success)
    """

    try:

        results = seg_model.predict(
            source=frame,
            verbose=False,
            device=DEVICE,
            conf=SEG_CONF,
        )

        if not results:
            return None, False

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return None, False

        if result.masks is None:
            return None, False

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        masks_xy = result.masks.xy

        if masks_xy is None:
            return None, False

        if len(masks_xy) == 0:
            return None, False

        # ----------------------------------------------------
        # Largest detected person
        # ----------------------------------------------------

        areas = (
            (boxes[:, 2] - boxes[:, 0])
            *
            (boxes[:, 3] - boxes[:, 1])
        )

        best_idx = int(
            np.argmax(areas)
        )

        x1, y1, x2, y2 = (
            boxes[best_idx]
        )

        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))

        x2 = min(
            frame.shape[1],
            int(np.ceil(x2))
        )

        y2 = min(
            frame.shape[0],
            int(np.ceil(y2))
        )

        if x2 <= x1 or y2 <= y1:
            return None, False

        # ----------------------------------------------------
        # Polygon mask
        # ----------------------------------------------------

        if best_idx >= len(masks_xy):
            return None, False

        polygon = masks_xy[best_idx]

        if polygon is None or len(polygon) < 3:
            return None, False

        mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        polygon_int = np.round(
            polygon
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [polygon_int],
            255
        )

        # ----------------------------------------------------
        # Morphological closing
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                MORPH_KERNEL_SIZE,
                MORPH_KERNEL_SIZE
            )
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ----------------------------------------------------
        # Largest connected component
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return None, False

        component_areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest_component_index = (
            int(np.argmax(component_areas))
            + 1
        )

        largest_area = int(
            stats[
                largest_component_index,
                cv2.CC_STAT_AREA
            ]
        )

        if largest_area < MIN_COMPONENT_AREA:
            return None, False

        clean_mask = np.where(
            labels == largest_component_index,
            255,
            0
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Bounding ROI around cleaned component
        # ----------------------------------------------------

        component_x = int(
            stats[
                largest_component_index,
                cv2.CC_STAT_LEFT
            ]
        )

        component_y = int(
            stats[
                largest_component_index,
                cv2.CC_STAT_TOP
            ]
        )

        component_w = int(
            stats[
                largest_component_index,
                cv2.CC_STAT_WIDTH
            ]
        )

        component_h = int(
            stats[
                largest_component_index,
                cv2.CC_STAT_HEIGHT
            ]
        )

        # ----------------------------------------------------
        # 10-pixel padded ROI
        # ----------------------------------------------------

        roi_x1 = max(
            0,
            component_x - ROI_PADDING
        )

        roi_y1 = max(
            0,
            component_y - ROI_PADDING
        )

        roi_x2 = min(
            frame.shape[1],
            component_x
            + component_w
            + ROI_PADDING
        )

        roi_y2 = min(
            frame.shape[0],
            component_y
            + component_h
            + ROI_PADDING
        )

        if (
            roi_x2 <= roi_x1
            or roi_y2 <= roi_y1
        ):
            return None, False

        crop = frame[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        crop_mask = clean_mask[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        if crop.size == 0:
            return None, False

        # ----------------------------------------------------
        # Apply segmentation mask
        # ----------------------------------------------------

        segmented = cv2.bitwise_and(
            crop,
            crop,
            mask=crop_mask
        )

        # ----------------------------------------------------
        # Aspect-preserving resize
        # ----------------------------------------------------

        scale = min(
            TARGET_W / segmented.shape[1],
            TARGET_H / segmented.shape[0]
        )

        new_w = max(
            1,
            int(
                segmented.shape[1]
                * scale
            )
        )

        new_h = max(
            1,
            int(
                segmented.shape[0]
                * scale
            )
        )

        resized = cv2.resize(
            segmented,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        resized_mask = cv2.resize(
            crop_mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Canvas
        # ----------------------------------------------------

        canvas = np.zeros(
            (
                TARGET_H,
                TARGET_W,
                3
            ),
            dtype=np.uint8
        )

        y_offset = (
            TARGET_H - new_h
        ) // 2

        x_offset = (
            TARGET_W - new_w
        ) // 2

        canvas[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Canny
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        return edges, True

    except Exception:
        return None, False


# ============================================================
# POSE EXTRACTION
# ============================================================

def extract_pose(
    frame,
    pose_model,
):
    """
    Extract the largest person's 17 COCO keypoints.

    Output:
        (17,3)
        x normalized to selected bbox
        y normalized to selected bbox
        confidence
    """

    try:

        results = pose_model.predict(
            source=frame,
            verbose=False,
            device=DEVICE,
            conf=0.25,
        )

        if not results:
            return None, False

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return None, False

        if result.keypoints is None:
            return None, False

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        keypoints = (
            result.keypoints
            .data
            .cpu()
            .numpy()
        )

        if len(keypoints) == 0:
            return None, False

        # ----------------------------------------------------
        # Largest detected person
        # ----------------------------------------------------

        areas = (
            (boxes[:, 2] - boxes[:, 0])
            *
            (boxes[:, 3] - boxes[:, 1])
        )

        best_idx = int(
            np.argmax(areas)
        )

        x1, y1, x2, y2 = (
            boxes[best_idx]
        )

        width = max(
            float(x2 - x1),
            1.0
        )

        height = max(
            float(y2 - y1),
            1.0
        )

        kp = keypoints[best_idx]

        output = np.zeros(
            (17, 3),
            dtype=np.float32
        )

        count = min(
            17,
            kp.shape[0]
        )

        for i in range(count):

            x = float(kp[i, 0])
            y = float(kp[i, 1])
            conf = float(kp[i, 2])

            output[i, 0] = (
                (x - x1) / width
            )

            output[i, 1] = (
                (y - y1) / height
            )

            output[i, 2] = conf

        return output, True

    except Exception:
        return None, False


# ============================================================
# TEMPORAL SAMPLING
# ============================================================

def choose_frame_indices(
    start_frame,
    end_frame,
    fps,
    tokens,
):
    """
    Select exactly `tokens` real source frames uniformly
    in physical time.

    GAVD frame numbers are 1-based.
    Returned indices are 0-based FFmpeg frame indices.
    """

    if end_frame < start_frame:
        return []

    start_zero = start_frame - 1
    end_zero = end_frame - 1

    if end_zero < start_zero:
        return []

    start_time = start_zero / fps
    end_time = end_zero / fps

    if tokens == 1:

        times = np.array(
            [start_time]
        )

    else:

        times = np.linspace(
            start_time,
            end_time,
            tokens
        )

    indices = np.rint(
        times * fps
    ).astype(np.int64)

    indices = np.clip(
        indices,
        start_zero,
        end_zero
    )

    indices = np.unique(
        indices
    )

    # Never duplicate frames just to reach 32 tokens.
    if len(indices) != tokens:
        return []

    return indices.tolist()


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_id,
    sequences,
    video_path,
    seg_model,
    pose_model,
):
    print("\n" + "=" * 80)
    print(f"VIDEO: {video_id}")
    print("=" * 80)

    info = get_video_info(
        video_path
    )

    fps = info["fps"]

    print(
        f"FPS: {fps:.6f}"
    )

    print(
        f"Frames: {info['frames']}"
    )

    print(
        f"Resolution: "
        f"{info['width']}x{info['height']}"
    )

    # --------------------------------------------------------
    # Build physical-time windows
    # --------------------------------------------------------

    all_required_frames = set()
    sequence_windows = []

    for seq in sequences:

        start_frame = int(
            seq["start_frame"]
        )

        end_frame = int(
            seq["end_frame"]
        )

        # GAVD frame numbers are 1-based.
        seq_start_time = (
            (start_frame - 1) / fps
        )

        seq_end_time = (
            (end_frame - 1) / fps
        )

        sequence_duration = (
            seq_end_time
            - seq_start_time
        )

        if sequence_duration < WINDOW_SECONDS:
            continue

        step = (
            WINDOW_SECONDS
            * (1.0 - OVERLAP)
        )

        window_start = seq_start_time

        while (
            window_start
            + WINDOW_SECONDS
            <= seq_end_time + 1e-6
        ):

            window_end = (
                window_start
                + WINDOW_SECONDS
            )

            # ------------------------------------------------
            # Convert physical-time boundaries back to
            # GAVD 1-based frame numbers.
            # ------------------------------------------------

            start_frame_window = max(
                start_frame,
                int(
                    np.ceil(
                        window_start * fps
                    )
                ) + 1,
            )

            end_frame_window = min(
                end_frame,
                int(
                    np.floor(
                        window_end * fps
                    )
                ) + 1,
            )

            if end_frame_window >= start_frame_window:

                indices = choose_frame_indices(
                    start_frame_window,
                    end_frame_window,
                    fps,
                    TOKENS_PER_WINDOW,
                )

                if len(indices) == TOKENS_PER_WINDOW:

                    all_required_frames.update(
                        indices
                    )

                    sequence_windows.append({
                        "sequence": seq,
                        "window_start": window_start,
                        "window_end": window_end,
                        "indices": indices,
                    })

            window_start += step

    print(
        f"Required unique source frames: "
        f"{len(all_required_frames)}"
    )

    print(
        f"Candidate windows: "
        f"{len(sequence_windows)}"
    )

    if not sequence_windows:

        return (
            0,
            0,
            0,
            0
        )

    # --------------------------------------------------------
    # Decode and extract features
    # --------------------------------------------------------

    feature_cache = {}

    def process_decoded_frame(
        zero_index,
        frame
    ):

        edge, edge_ok = (
            extract_person_segmentation(
                frame,
                seg_model
            )
        )

        pose, pose_ok = (
            extract_pose(
                frame,
                pose_model
            )
        )

        feature_cache[
            zero_index
        ] = {
            "edge": edge,
            "pose": pose,
            "edge_ok": edge_ok,
            "pose_ok": pose_ok,
        }

    try:

        decoded_total, required_decoded = (
            stream_ffmpeg_frames(
                video_path,
                sorted(all_required_frames),
                info["width"],
                info["height"],
                process_decoded_frame,
            )
        )

    except Exception as e:

        print(
            f"  FFmpeg error: {e}"
        )

        return (
            0,
            0,
            0,
            len(all_required_frames)
        )

    print(
        f"  Feature cache: "
        f"{len(feature_cache)}/"
        f"{len(all_required_frames)} required frames"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_SEQUENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = 0
    seg_failures = 0
    pose_failures = 0
    decode_failures = 0

    # --------------------------------------------------------
    # Process windows
    # --------------------------------------------------------

    for (
        window_number,
        window
    ) in enumerate(
        sequence_windows
    ):

        seq = window["sequence"]
        indices = window["indices"]

        edges = np.zeros(
            (
                TOKENS_PER_WINDOW,
                TARGET_H,
                TARGET_W
            ),
            dtype=np.uint8
        )

        keypoints = np.zeros(
            (
                TOKENS_PER_WINDOW,
                17,
                3
            ),
            dtype=np.float32
        )

        frame_numbers = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.int32
        )

        timestamps = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.float64
        )

        relative_timestamps = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.float32
        )

        delta_t = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.float32
        )

        # New audit masks: explicitly tell the model/audit
        # which tokens contain successfully extracted features.
        segmentation_valid = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.uint8
        )

        pose_valid = np.zeros(
            TOKENS_PER_WINDOW,
            dtype=np.uint8
        )

        valid_start_time = float(
            window["window_start"]
        )

        for i, zero_index in enumerate(
            indices
        ):

            # Convert 0-based index back to GAVD 1-based frame.
            frame_numbers[i] = (
                zero_index + 1
            )

            timestamp = (
                zero_index / fps
            )

            timestamps[i] = timestamp

            relative_timestamps[i] = (
                timestamp
                - valid_start_time
            )

            if i > 0:

                delta_t[i] = (
                    timestamp
                    - timestamps[i - 1]
                )

            features = feature_cache.get(
                zero_index
            )

            if features is None:

                decode_failures += 1
                continue

            edge = features["edge"]
            pose = features["pose"]

            if (
                features["edge_ok"]
                and edge is not None
            ):

                edges[i] = edge
                segmentation_valid[i] = 1

            else:

                seg_failures += 1

            if (
                features["pose_ok"]
                and pose is not None
            ):

                keypoints[i] = pose
                pose_valid[i] = 1

            else:

                pose_failures += 1

        # ----------------------------------------------------
        # Save NPZ
        # ----------------------------------------------------

        sequence_id = seq[
            "sequence_id"
        ]

        gait_label = seq[
            "gait_pat"
        ]

        cam_view = seq[
            "cam_view"
        ]

        filename = (
            f"{video_id}__"
            f"{sequence_id}__"
            f"{window_number:05d}.npz"
        )

        output_path = (
            OUTPUT_SEQUENCE_DIR
            / filename
        )

        np.savez_compressed(
            output_path,

            edges=edges,

            keypoints=keypoints,

            frame_numbers=frame_numbers,

            timestamps=timestamps,

            relative_timestamps=(
                relative_timestamps
            ),

            delta_t=delta_t,

            source_fps=np.float32(
                fps
            ),

            video_id=video_id,

            sequence_id=sequence_id,

            gait_label=gait_label,

            cam_view=cam_view,

            start_time=np.float64(
                window["window_start"]
            ),

            end_time=np.float64(
                window["window_end"]
            ),

            # Explicit feature-validity masks.
            segmentation_valid=(
                segmentation_valid
            ),

            pose_valid=pose_valid,
        )

        generated += 1

    print(
        f"Generated windows: {generated}"
    )

    print(
        f"Segmentation failures: "
        f"{seg_failures}"
    )

    print(
        f"Pose failures: "
        f"{pose_failures}"
    )

    print(
        f"Decode failures: "
        f"{decode_failures}"
    )

    return (
        generated,
        seg_failures,
        pose_failures,
        decode_failures,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "GAVD V2 FULL-DATASET PREPROCESSING"
    )
    print(
        "FFmpeg + V1-style segmentation + "
        "physical-time sampling"
    )
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Window: {WINDOW_SECONDS} seconds"
    )

    print(
        f"Overlap: {OVERLAP * 100:.0f}%"
    )

    print(
        f"Tokens: {TOKENS_PER_WINDOW}"
    )

    print(
        "\nSpatial preprocessing:"
    )

    print(
        "  YOLO11n segmentation"
    )

    print(
        "  Largest person"
    )

    print(
        "  Polygon mask"
    )

    print(
        f"  Morphological closing: "
        f"{MORPH_KERNEL_SIZE}x"
        f"{MORPH_KERNEL_SIZE}"
    )

    print(
        f"  Minimum component area: "
        f"{MIN_COMPONENT_AREA}"
    )

    print(
        f"  ROI padding: "
        f"{ROI_PADDING}px"
    )

    print(
        "  Aspect-preserving resize"
    )

    print(
        f"  Output: "
        f"{TARGET_H}x{TARGET_W}"
    )

    print(
        "  Canny: 50,150"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_SEQUENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "\nLoading YOLO segmentation model..."
    )

    seg_model = YOLO(
        YOLO_SEG_MODEL
    )

    print(
        "Loading YOLO pose model..."
    )

    pose_model = YOLO(
        YOLO_POSE_MODEL
    )

    # --------------------------------------------------------
    # Load accessible videos
    # --------------------------------------------------------

    available_video_ids = (
        load_available_video_ids()
    )

    print(
        f"\nAccessible videos according to "
        f"availability audit: "
        f"{len(available_video_ids)}"
    )

    # --------------------------------------------------------
    # Load sequence annotations
    # --------------------------------------------------------

    sequence_rows = (
        load_sequence_summary()
    )

    print(
        f"Total annotated sequences: "
        f"{len(sequence_rows)}"
    )

    grouped = {}

    for row in sequence_rows:

        video_id = row[
            "video_id"
        ]

        if video_id not in available_video_ids:
            continue

        grouped.setdefault(
            video_id,
            []
        ).append(row)

    print(
        f"Videos with annotations: "
        f"{len(grouped)}"
    )

    print(
        f"Sequences for accessible videos: "
        f"{sum(len(v) for v in grouped.values())}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR / "metadata.csv"
    )

    metadata_fields = [
        "filename",
        "video_id",
        "sequence_id",
        "gait_pat",
        "cam_view",
        "window_start",
        "window_end",
        "source_fps",
        "tokens",
        "segmentation_valid_tokens",
        "pose_valid_tokens",
    ]

    metadata_file = open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    metadata_writer = csv.DictWriter(
        metadata_file,
        fieldnames=metadata_fields
    )

    metadata_writer.writeheader()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_windows = 0
    total_seg_failures = 0
    total_pose_failures = 0
    total_decode_failures = 0

    videos_generated = 0
    videos_available = 0

    # --------------------------------------------------------
    # Process all accessible videos
    # --------------------------------------------------------

    video_ids = sorted(
        grouped.keys()
    )

    print(
        f"\nVideos scheduled for preprocessing: "
        f"{len(video_ids)}"
    )

    for video_number, video_id in enumerate(
        video_ids,
        start=1
    ):

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"VIDEO {video_number}/"
            f"{len(video_ids)}: "
            f"{video_id}"
        )

        print(
            "#" * 80
        )

        video_path = (
            VIDEO_DIR
            / f"{video_id}.mp4"
        )

        if not video_path.exists():

            print(
                f"SKIPPING {video_id}: "
                f"video not found"
            )

            continue

        videos_available += 1

        sequences = grouped.get(
            video_id,
            []
        )

        if not sequences:

            print(
                f"SKIPPING {video_id}: "
                f"no annotations"
            )

            continue

        try:

            (
                generated,
                seg_fail,
                pose_fail,
                decode_fail,
            ) = process_video(
                video_id,
                sequences,
                video_path,
                seg_model,
                pose_model,
            )

            # ------------------------------------------------
            # Write metadata for generated NPZs
            # ------------------------------------------------

            video_npz_files = sorted(
                OUTPUT_SEQUENCE_DIR.glob(
                    f"{video_id}__*.npz"
                )
            )

            for npz_path in video_npz_files:

                with np.load(
                    npz_path,
                    allow_pickle=True
                ) as data:

                    metadata_writer.writerow({
                        "filename":
                            npz_path.name,

                        "video_id":
                            str(
                                data[
                                    "video_id"
                                ]
                            ),

                        "sequence_id":
                            str(
                                data[
                                    "sequence_id"
                                ]
                            ),

                        "gait_pat":
                            str(
                                data[
                                    "gait_label"
                                ]
                            ),

                        "cam_view":
                            str(
                                data[
                                    "cam_view"
                                ]
                            ),

                        "window_start":
                            float(
                                data[
                                    "start_time"
                                ]
                            ),

                        "window_end":
                            float(
                                data[
                                    "end_time"
                                ]
                            ),

                        "source_fps":
                            float(
                                data[
                                    "source_fps"
                                ]
                            ),

                        "tokens":
                            TOKENS_PER_WINDOW,

                        "segmentation_valid_tokens":
                            int(
                                np.sum(
                                    data[
                                        "segmentation_valid"
                                    ]
                                )
                            ),

                        "pose_valid_tokens":
                            int(
                                np.sum(
                                    data[
                                        "pose_valid"
                                    ]
                                )
                            ),
                    })

            metadata_file.flush()

            if generated > 0:
                videos_generated += 1

            total_windows += generated

            total_seg_failures += seg_fail
            total_pose_failures += pose_fail
            total_decode_failures += decode_fail

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{video_id}:"
            )

            print(
                repr(e)
            )

    metadata_file.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "videos_selected":
            len(video_ids),

        "videos_available":
            videos_available,

        "videos_with_generated_data":
            videos_generated,

        "windows":
            total_windows,

        "tokens_per_window":
            TOKENS_PER_WINDOW,

        "window_seconds":
            WINDOW_SECONDS,

        "overlap":
            OVERLAP,

        "edge_shape":
            [
                TOKENS_PER_WINDOW,
                TARGET_H,
                TARGET_W,
            ],

        "keypoint_shape":
            [
                TOKENS_PER_WINDOW,
                17,
                3,
            ],

        "segmentation_failures":
            total_seg_failures,

        "pose_failures":
            total_pose_failures,

        "decode_failures":
            total_decode_failures,

        "decoder":
            "FFmpeg/libdav1d",

        "interpolation":
            False,

        "temporal_representation":
            (
                "physical-time windows "
                "with original source "
                "timestamps"
            ),

        "spatial_representation":
            (
                "V1-style YOLO segmentation "
                "with polygon mask, "
                "5x5 morphological closing, "
                "largest connected component, "
                "500-pixel minimum area, "
                "10-pixel ROI padding, "
                "aspect-preserving resize, "
                "and Canny(50,150)"
            ),

        "segmentation_model":
            YOLO_SEG_MODEL,

        "pose_model":
            YOLO_POSE_MODEL,

        "source_dataset_videos":
            348,

        "accessible_videos":
            len(available_video_ids),

        "output_directory":
            str(OUTPUT_DIR),
    }

    summary_path = (
        OUTPUT_DIR / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print(
        "\nMetadata:"
    )

    print(
        metadata_path
    )

    print(
        "\nNPZ directory:"
    )

    print(
        OUTPUT_SEQUENCE_DIR
    )


if __name__ == "__main__":
    main()
