import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_ROOT = BASE_DIR / "pipeline_outputs"

sys.path.insert(0, str(BASE_DIR))

from predict import predict
from evidence import build_evidence


def load_yolo_extractor():
    script = PROJECT_DIR / "gaitxplain_inference.py"

    spec = importlib.util.spec_from_file_location(
        "gaitxplain_yolo_extractor", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if width <= 0 or height <= 0 or frames < 32:
        raise ValueError(
            "Video must have valid dimensions and at least 32 frames."
        )

    return fps, width, height, frames


def format_local_evidence(report, evidence_json, recording_id):
    prediction_data = report["video_prediction"]
    biomechanical = evidence_json.get(
        "biomechanical_evidence", {}
    )

    lines = [
        "CURRENT VIDEO VERIFIED EVIDENCE",
        f"Recording ID: {recording_id}",
        f"Frames processed: {report.get('frames_processed')}",
        f"Windows evaluated: {report.get('sequences_evaluated')}",
        "",
        "AUTHORITATIVE MODEL PREDICTION",
        f"Binary classification: {prediction_data.get('binary_class')}",
        f"Normal probability: {prediction_data.get('normal_probability')}",
        f"Abnormal probability: {prediction_data.get('abnormal_probability')}",
        f"Seven-class prediction: {prediction_data.get('gait_class')}",
        "",
        "WINDOW ANALYSIS",
        json.dumps(
            evidence_json.get("window_analysis", {}),
            indent=2
        ),
        "",
        "FEATURE STATISTICS FROM THIS VIDEO"
    ]

    feature_summary = report.get("feature_summary", {})
    features = feature_summary.get("features", {})

    for name, statistics in features.items():
        lines.append(
            f"{name}: {json.dumps(statistics, ensure_ascii=False)}"
        )

    if not features:
        lines.append("No feature statistics available.")

    lines.extend([
        "",
        "MEASUREMENT LIMITATIONS",
        str(biomechanical.get("interpretation_note", "")),
        "Features are image-coordinate-derived, not calibrated physical measurements."
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="GaitXplain complete video analysis pipeline"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input walking video"
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    recording_id = video_path.stem
    output_dir = OUTPUT_ROOT / recording_id
    pose_dir = output_dir / "pose"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GAITXPLAIN COMPLETE ANALYSIS")
    print("=" * 60)

    print("\n[1/5] Reading video metadata...")
    fps, width, height, frames = get_video_metadata(video_path)

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frames}")

    print("\n[2/5] Extracting YOLO pose keypoints...")
    yolo_module = load_yolo_extractor()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    yolo_model = YOLO(str(yolo_module.MODEL_PATH))

    keypoints_csv, annotated_video = yolo_module.extract_pose(
        video_path,
        yolo_model,
        pose_dir,
        device
    )

    del yolo_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[3/5] Running trained GaitXplain classifier...")
    prediction_dir = output_dir / "prediction"

    predict(
        str(keypoints_csv),
        str(prediction_dir),
        width,
        height
    )

    report_path = prediction_dir / "prediction_report.json"

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    print("\n[4/5] Generating biomechanical evidence...")
    evidence = build_evidence(report_path)

    evidence_path = output_dir / "explanation_evidence.json"

    with evidence_path.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, indent=4)

    print("Evidence saved:", evidence_path)

    print("\n[5/5] Generating grounded Qwen explanation...")

    # Importing this module opens the existing ChromaDB collection.
    # It does not rebuild the database.
    import gaitxplain_rag_llm as rag

    # The project's global interpretation constraints are loaded
    # from the existing RAG collection.
    question = (
        "Write a concise evidence-based explanation in three sections. "
        "First, state the authoritative final model classification and "
        "its model probability, clearly identifying it as an uncalibrated "
        "model output. Second, describe the actual joint-angle and "
        "left-right asymmetry statistics available for THIS recording, "
        "using their exact mean and standard deviation values from the "
        "evidence. Include only a few relevant observations; do not list "
        "all feature names. Third, state the limitations. "
        "Do not claim that the features caused the classification, "
        "do not infer a medical diagnosis, and do not describe a Normal "
        "model output as proof of clinical normality. "
        "If a requested statistic is unavailable, say so. "
        f"Recording ID: {recording_id}"
    )

    metadata = {
        "binary_class": report["video_prediction"]["binary_class"],
        "gait_class": report["video_prediction"]["gait_class"]
    }

    prediction_context = {"metadata": metadata}

    final_classification = rag.determine_final_classification(
        metadata
    )

    local_evidence = format_local_evidence(
        report,
        evidence,
        recording_id
    )

    # Retrieve general background from the existing RAG database.
    # These retrieved recordings are not treated as measurements
    # belonging to the current video.
    rag_context = ""
    rag_sources = []

    try:
        retrieved, sources, _, _ = rag.retrieve_evidence(
            "Project-wide gait interpretation constraints and limitations",
            None
        )

        rag_sources = sources or []

        # Only include general constraint documents, never another
        # recording's prediction or feature statistics.
        for source in rag_sources:
            chunk_type = str(source.get("chunk_type", "")).lower()

            if "constraint" in chunk_type or "global" in chunk_type:
                rag_context += (
                    "\nRAG PROJECT-WIDE REFERENCE:\n"
                    + str(rag.GLOBAL_CONSTRAINTS)
                )
                break

    except Exception as error:
        print("RAG retrieval warning:", error)

    combined_evidence = local_evidence

    if rag_context:
        combined_evidence += "\n\n" + rag_context

    answer = rag.generate_explanation(
        question,
        combined_evidence,
        recording_id,
        prediction_context
    )

    explanation_path = output_dir / "gaitxplain_explanation.txt"

    with explanation_path.open("w", encoding="utf-8") as file:
        file.write("GAITXPLAIN ANALYSIS\n")
        file.write("=" * 50 + "\n")
        file.write(f"Recording: {recording_id}\n")
        file.write(f"Final classification: {final_classification}\n\n")
        file.write(answer)

    print("\n" + "=" * 60)
    print("GAITXPLAIN FINAL RESULT")
    print("=" * 60)
    print("Recording:", recording_id)
    print("Final classification:", final_classification)
    print("\nQWEN EXPLANATION:\n")
    print(answer)

    print("\nOUTPUT FILES")
    print("Prediction:", report_path)
    print("Evidence:", evidence_path)
    print("Explanation:", explanation_path)
    print("Annotated video:", annotated_video)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
