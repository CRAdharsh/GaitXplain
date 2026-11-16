import cv2
import csv
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"C:\Users\ASUS\Downloads\finla project dataset")

MODEL_PATH = PROJECT / "mam's dataset formation" / "yolo11n-pose.pt"

OUTPUT_DIR = (
    PROJECT / "GaitXplain_Pose_Experiments" / "Inference_Test"
)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


def extract_pose(video_path, model, output_dir, device):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video metadata")

    name = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}_keypoints.csv"
    video_output = output_dir / f"{name}_annotated.mp4"

    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create annotated video")

    headers = ["frame", "person_detected"]

    for joint in KEYPOINT_NAMES:
        headers.extend([
            f"{joint}_x",
            f"{joint}_y",
            f"{joint}_confidence"
        ])

    frame_number = 0
    detected_frames = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.25,
                device=device,
                verbose=False
            )

            result = results[0]

            row = [frame_number, 0]

            if result.keypoints is not None and len(result.keypoints) > 0:

                if result.boxes is not None and len(result.boxes) > 0:
                    scores = result.boxes.conf.cpu().numpy()
                    best = int(scores.argmax())
                else:
                    best = 0

                points = result.keypoints.xy[best].cpu().numpy()

                if result.keypoints.conf is not None:
                    confidence = (
                        result.keypoints.conf[best].cpu().numpy()
                    )
                else:
                    confidence = [0.0] * len(KEYPOINT_NAMES)

                row[1] = 1
                detected_frames += 1

                for i in range(len(KEYPOINT_NAMES)):
                    x, y = points[i]

                    row.extend([
                        round(float(x), 3),
                        round(float(y), 3),
                        round(float(confidence[i]), 4)
                    ])

            else:
                row.extend([None] * (len(KEYPOINT_NAMES) * 3))

            csv_writer.writerow(row)

            annotated = result.plot()
            writer.write(annotated)

            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames")

    cap.release()
    writer.release()

    detection_rate = (
        detected_frames / frame_number * 100
        if frame_number else 0
    )

    print("\n===== POSE EXTRACTION RESULT =====")
    print("Video:", name)
    print("FPS:", fps)
    print("Resolution:", width, "x", height)
    print("Frames:", frame_number)
    print("Detected frames:", detected_frames)
    print("Detection rate:", round(detection_rate, 2), "%")
    print("Keypoints CSV:", csv_path)
    print("Annotated video:", video_output)

    return csv_path, video_output


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Selected device:", device)
    print("Loading YOLO11 pose model...")

    model = YOLO(str(MODEL_PATH))

    # First test using an existing sample video.
    test_video = Path(
    r"C:\Users\ASUS\Downloads\mam_approach_extracted\mam approach dataset\Normal\Normal_1.mp4"
)

    # If the above location does not exist, update this path
    # to the actual Normal_1.mp4 location.

    extract_pose(test_video, model, OUTPUT_DIR, device)


if __name__ == "__main__":
    main()
