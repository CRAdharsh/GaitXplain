import re

import json

import urllib.request

import urllib.error

from pathlib import Path



import chromadb

import torch

from sentence_transformers import SentenceTransformer





# ============================================================

# CONFIGURATION

# ============================================================



BASE_DIR = Path(__file__).resolve().parent

DB_DIR = BASE_DIR / "video_rag_db"



COLLECTION_NAME = "gaitxplain_video_evidence_v2"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



OLLAMA_URL = "http://localhost:11434/api/chat"

LLM_MODEL = "qwen2.5:3b"



TOP_K = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





# ============================================================

# FEATURE ROUTING

# ============================================================



FEATURE_ROUTES = {

    "joint_angles": [

        "joint angle", "knee angle", "hip angle",

        "angle statistics", "angles"

    ],

    "left_right_asymmetry": [

        "asymmetry", "left right difference",

        "side difference", "bilateral difference"

    ],

    "joint_angular_velocity": [

        "angular velocity", "rotation velocity"

    ],

    "limb_segment_lengths": [

        "limb length", "thigh length",

        "shank length", "segment length"

    ],

    "joint_linear_velocity": [

        "linear velocity", "joint velocity"

    ],

    "joint_acceleration": [

        "acceleration", "joint acceleration"

    ]

}





def detect_feature_group(question):

    q = question.lower()



    for group, keywords in FEATURE_ROUTES.items():

        if any(word in q for word in keywords):

            return group



    return None





# ============================================================

# CONNECT TO EXISTING DATABASE

# ============================================================



print("\nLoading GaitXplain RAG V2...")



client = chromadb.PersistentClient(path=str(DB_DIR))



collection = client.get_collection(

    name=COLLECTION_NAME

)



print("Collection:", COLLECTION_NAME)

print("Documents:", collection.count())





# ============================================================

# LOAD EMBEDDING MODEL

# ============================================================



print("\nLoading embedding model on", DEVICE)



embedding_model = SentenceTransformer(

    EMBEDDING_MODEL,

    device=DEVICE

)



print("Embedding dimension:",

      embedding_model.get_embedding_dimension())





# ============================================================

# GLOBAL CONSTRAINTS

# ============================================================



def get_global_constraints():



    result = collection.get(

        where={"recording_id": "PROJECT"},

        include=["documents"]

    )



    return "\n".join(result.get("documents", []))





GLOBAL_CONSTRAINTS = get_global_constraints()





# ============================================================

# GET AUTHORITATIVE PREDICTION SUMMARY

# ============================================================



def get_prediction_summary(recording_id):



    result = collection.get(

        where={

            "$and": [

                {"recording_id": recording_id},

                {"chunk_type": "prediction_summary"}

            ]

        },

        include=["documents", "metadatas"]

    )



    if not result["documents"]:

        return None



    return {

        "document": result["documents"][0],

        "metadata": result["metadatas"][0]

    }





def determine_final_classification(metadata):



    binary_class = metadata.get("binary_class", "Unknown")

    gait_class = metadata.get("gait_class", "None")



    if binary_class == "Normal":

        return "Normal"



    if binary_class == "Abnormal":

        if gait_class and gait_class != "None":

            return gait_class

        return "Abnormal (category unavailable)"



    return "Unknown"





# ============================================================

# RETRIEVE QUESTION-SPECIFIC EVIDENCE

# ============================================================



def retrieve_evidence(question, recording_id=None):



    embedding = embedding_model.encode(

        [question],

        convert_to_numpy=True,

        normalize_embeddings=True

    )



    feature_group = detect_feature_group(question)



    search_args = {

        "query_embeddings": embedding.tolist(),

        "n_results": TOP_K,

        "include": ["documents", "metadatas", "distances"]

    }



    filters = []



    if recording_id:

        filters.append({"recording_id": recording_id})



    if feature_group:

        filters.append({

            "chunk_type": f"features_{feature_group}"

        })



    if len(filters) == 1:

        search_args["where"] = filters[0]

    elif len(filters) > 1:

        search_args["where"] = {"$and": filters}



    results = collection.query(**search_args)



    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]



    # Fallback if a feature-specific query has no results.

    if not documents and feature_group:



        print("No feature-specific result. Trying general retrieval...")



        fallback_args = {

            "query_embeddings": embedding.tolist(),

            "n_results": TOP_K,

            "include": ["documents", "metadatas", "distances"]

        }



        if recording_id:

            fallback_args["where"] = {

                "recording_id": recording_id

            }



        results = collection.query(**fallback_args)



        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        distances = results["distances"][0]



    evidence_blocks = []

    sources = []



    # Add the authoritative prediction summary separately.

    prediction = None



    if recording_id:

        prediction = get_prediction_summary(recording_id)



        if prediction:



            metadata = prediction["metadata"]



            binary_class = metadata.get("binary_class", "Unknown")

            gait_class = metadata.get("gait_class", "None")

            final_class = determine_final_classification(metadata)



            evidence_blocks.append(

                f"""

AUTHORITATIVE PREDICTION SUMMARY



Recording: {recording_id}

Binary classification: {binary_class}

Seven-class prediction: {gait_class}

Final classification: {final_class}



Original prediction evidence:

{prediction['document']}

"""

            )



            sources.append({

                "recording_id": recording_id,

                "chunk_type": "prediction_summary",

                "final_classification": final_class,

                "cosine_distance": "authoritative lookup"

            })



    # Add the retrieved feature or other evidence.

    for i, document in enumerate(documents):



        metadata = metadatas[i]



        rid = metadata.get("recording_id", "Unknown")

        chunk = metadata.get("chunk_type", "Unknown")



        final_class = metadata.get(

            "final_classification", "Not specified"

        )



        # Avoid duplicating the prediction summary.

        if chunk == "prediction_summary":

            continue



        evidence_blocks.append(

            f"""

RETRIEVED EVIDENCE



Recording: {rid}

Evidence type: {chunk}

Final classification in source metadata: {final_class}



{document}

"""

        )



        sources.append({

            "recording_id": rid,

            "chunk_type": chunk,

            "final_classification": final_class,

            "cosine_distance": round(float(distances[i]), 4)

        })



    return (

        "\n".join(evidence_blocks),

        sources,

        feature_group,

        prediction

    )





# ============================================================

# GENERATE GROUNDED EXPLANATION

# ============================================================



def generate_explanation(

    question,

    evidence,

    recording_id,

    prediction

):



    if not evidence.strip():

        return "No relevant evidence was retrieved."



    classification_facts = "Not available."



    if prediction:



        metadata = prediction["metadata"]



        binary_class = metadata.get("binary_class", "Unknown")

        gait_class = metadata.get("gait_class", "None")

        final_class = determine_final_classification(metadata)



        classification_facts = f"""

Recording ID: {recording_id}

Binary classification: {binary_class}

Seven-class prediction: {gait_class}

Final classification: {final_class}



These values are authoritative database metadata.

Do not reinterpret or change them.

"""



    # Deterministically identify feature keys actually present in retrieved evidence.

    available_features = []



    for line in evidence.splitlines():

        line = line.strip()



        if (
            line.startswith("temporal_")
            or line.startswith(("left_", "right_"))
        ) and ":" in line:

            feature_name = line.split(":", 1)[0].strip()



            if feature_name not in available_features:

                available_features.append(feature_name)



    if available_features:

        feature_availability = "\\n".join(

            f"- {feature}" for feature in available_features

        )

    else:

        feature_availability = "No individual feature keys were identified."



    system_prompt = """

You are GaitXplain, an evidence-grounded gait analysis assistant.



Follow these rules strictly:



CLASSIFICATION:

1. Use the authoritative classification facts supplied

   by the system.

2. Never invent or change the binary classification.

3. Binary Normal means final classification Normal.

4. Binary Abnormal means the seven-class prediction

   supplies the final category when available.

5. Never say a binary output is Ataxic, Parkinsonian,

   or another seven-class category.



EVIDENCE:

6. Use only supplied evidence for recording-specific facts.

7. Never invent numerical values, measurements, feature names,

   joint names, or statistics.

8. Distinguish prediction outputs from feature statistics.

9. Do not claim that feature values explain why a classifier

   made its decision unless the evidence establishes this.

10. Report only the exact features and joint names present

    in the retrieved evidence.

11. Never copy, reuse, infer, or substitute one joint's

    statistics for another joint.

12. If the question asks about a feature that is absent from

    the evidence, explicitly state that it is unavailable.

13. Do not assume that hip, knee, and ankle measurements

    all exist just because the question mentions them.

14. Preserve the original numerical values from the evidence.

    Do not calculate replacement values or fabricate missing

    statistics.

15. If evidence is ambiguous, explain the ambiguity instead

    of guessing.



MEASUREMENT LIMITATIONS:

11. Features are derived from image coordinates.

12. They are not calibrated physical measurements or

    clinically validated gait measurements.

13. Limb segment asymmetry values must not be described

    in degrees unless the source explicitly specifies degrees.

14. Model probabilities are not calibrated diagnostic confidence.



SAFETY:

15. Do not diagnose patients or prescribe treatment.

16. Do not claim clinical validation.



STYLE:

17. Answer directly and avoid repetition.

18. Use short headings and concise bullet points when useful.

19. Explain mean and standard deviation simply when asked.

20. Finish prediction explanations with a brief limitation note.

"""



    user_prompt = f"""

USER QUESTION:

{question}



AUTHORITATIVE CLASSIFICATION FACTS:

{classification_facts}



RETRIEVED EVIDENCE:

{evidence}



VERIFIED FEATURE AVAILABILITY:

{feature_availability}



Only report statistics for feature keys listed under VERIFIED FEATURE AVAILABILITY.

If a joint or feature is absent from that list, explicitly say its statistics

are unavailable. Never copy statistics from another joint.



PROJECT-WIDE CONSTRAINTS:

{GLOBAL_CONSTRAINTS}



Answer using only the retrieved evidence and verified feature availability.

Keep the answer concise and do not repeat the same point.

"""



    payload = {

        "model": LLM_MODEL,

        "messages": [

            {"role": "system", "content": system_prompt},

            {"role": "user", "content": user_prompt}

        ],

        "stream": False,

        "options": {

            "temperature": 0.1

        }

    }



    request = urllib.request.Request(

        OLLAMA_URL,

        data=json.dumps(payload).encode("utf-8"),

        headers={"Content-Type": "application/json"},

        method="POST"

    )



    try:



        with urllib.request.urlopen(

            request,

            timeout=180

        ) as response:



            result = json.loads(

                response.read().decode("utf-8")

            )





        answer = result["message"]["content"]

        # RESPONSE QUALITY CHECK: reject feature inventories as explanations.
        answer_lines = [line.strip() for line in answer.splitlines() if line.strip()]

        feature_line_count = sum(
            any(token in line.lower() for token in (
                "temporal_", "left_", "right_"
            ))
            for line in answer_lines
        )

        inventory_response = (
            feature_line_count >= 4
            or "available features:" in answer.lower()
            or "features have valid statistics" in answer.lower()
        )

        if inventory_response:
            print(
                "\\n[RESPONSE QUALITY CHECK] Feature inventory detected. "
                "Returning verified video evidence instead."
            )

            verified_lines = [
                line.strip()
                for line in evidence.splitlines()
                if line.strip().startswith((
                    "temporal_", "left_", "right_"
                ))
                and ":" in line
            ]

            probability_lines = [
                line.strip()
                for line in evidence.splitlines()
                if "probability:" in line.lower()
            ]

            clean_evidence = evidence.replace("\\n", "\n")
            clean_classification = classification_facts.replace("\\n", "\n")

            probabilities = {}
            for line in clean_evidence.splitlines():
                if ":" not in line:
                    continue
                label, value = line.split(":", 1)
                label = label.strip().lower()
                if label in ("normal probability", "abnormal probability"):
                    probabilities[label] = value.strip()

            selected_features = (
                "left_hip_angle",
                "right_hip_angle",
                "left_knee_angle",
                "right_knee_angle"
            )

            verified_stats = {}
            for line in clean_evidence.splitlines():
                if ":" not in line:
                    continue
                name, raw_stats = line.split(":", 1)
                name = name.strip()
                if name not in selected_features:
                    continue
                try:
                    stats = json.loads(raw_stats.strip())
                    if "mean" in stats and "std" in stats:
                        verified_stats[name] = stats
                except (ValueError, TypeError):
                    continue

            fallback_lines = [
                "GAITXPLAIN VIDEO ANALYSIS",
                "",
                "CLASSIFICATION",
                clean_classification.strip(),
                ""
            ]

            for label in ("normal probability", "abnormal probability"):
                if label in probabilities:
                    fallback_lines.append(
                        f"{label.title()}: {float(probabilities[label]):.4%}"
                    )

            fallback_lines.extend([
                "",
                "JOINT ANGLE SUMMARY",
            ])

            display_names = {
                "left_hip_angle": "Left hip",
                "right_hip_angle": "Right hip",
                "left_knee_angle": "Left knee",
                "right_knee_angle": "Right knee"
            }

            for name in selected_features:
                stats = verified_stats.get(name)
                if stats is not None:
                    fallback_lines.append(
                        f"{display_names[name]}: "
                        f"{stats['mean']:.2f} degrees "
                        f"(SD {stats['std']:.2f})"
                    )

            if not verified_stats:
                fallback_lines.append(
                    "Joint-angle statistics were not available."
                )

            # Three-layer biomechanical interpretation
            hip_l = verified_stats.get("left_hip_angle")
            hip_r = verified_stats.get("right_hip_angle")
            knee_l = verified_stats.get("left_knee_angle")
            knee_r = verified_stats.get("right_knee_angle")

            interpretation_lines = [
                "",
                "LAYER 1 - OBSERVATIONS",
            ]

            if hip_l and hip_r:
                hip_diff = abs(float(hip_l["mean"]) - float(hip_r["mean"]))
                interpretation_lines.append(
                    f"The mean left-right hip angle difference is "
                    f"{hip_diff:.2f} degrees."
                )

            if knee_l and knee_r:
                knee_diff = abs(float(knee_l["mean"]) - float(knee_r["mean"]))
                interpretation_lines.append(
                    f"The mean left-right knee angle difference is "
                    f"{knee_diff:.2f} degrees."
                )

            if not (hip_l and hip_r and knee_l and knee_r):
                interpretation_lines.append(
                    "Some bilateral joint-angle measurements were unavailable."
                )

            interpretation_lines.extend([
                "",
                "LAYER 2 - BIOMECHANICAL INTERPRETATION",
            ])

            variability = []
            for label, left, right in (
                ("hip", hip_l, hip_r),
                ("knee", knee_l, knee_r)
            ):
                if left and right:
                    left_sd = float(left["std"])
                    right_sd = float(right["std"])
                    variability.append((label, left_sd, right_sd))

            if variability:
                for label, left_sd, right_sd in variability:
                    if abs(left_sd - right_sd) < 0.01:
                        description = "similar measured variability on both sides"
                    elif left_sd > right_sd:
                        description = "greater measured variability on the left side"
                    else:
                        description = "greater measured variability on the right side"

                    interpretation_lines.append(
                        f"The {label} angle standard deviations indicate "
                        f"{description} across the analysed frames "
                        f"(left SD: {left_sd:.2f}, right SD: {right_sd:.2f})."
                    )

                interpretation_lines.append(
                    "These standard deviations describe variation in the "
                    "extracted angle measurements; they do not by themselves "
                    "establish irregular or impaired walking."
                )

            interpretation_lines.extend([
                "",
                "LAYER 3 - CLASSIFICATION RELEVANCE",
                "The model's classification is based on the combined "
                "32-feature representation and temporal information. "
                "The joint-angle observations provide additional descriptive "
                "context, but no individual angle or bilateral difference "
                "independently proves the predicted gait category.",
                "",
                "REFERENCE AND LIMITATION",
                "No validated reference range matched for camera setup, "
                "walking conditions and measurement method was supplied. "
                "Therefore, these values cannot be labelled higher or lower "
                "than an expected clinical range."
            ])

            fallback_lines.extend(interpretation_lines)
            fallback_lines.extend([
                "",
                "GENERAL LIMITATIONS",
                "Measurements are image-coordinate-derived, not calibrated "
                "physical measurements. The model was trained on a small "
                "dataset and has not established clinical validity or "
                "independent-subject generalization. This output is not "
                "a medical diagnosis."
            ])

            return "\n".join(fallback_lines)




        # ----------------------------------------------------

        # DETERMINISTIC ANSWER VALIDATION

        # ----------------------------------------------------



        answer_lower = answer.lower()



        # Identify which joints are actually supported.

        available_joints = set()



        for feature in available_features:

            feature_lower = feature.lower()



            for joint in ("hip", "knee", "ankle"):

                if joint in feature_lower.split("_"):

                    available_joints.add(joint)



        mentioned_joints = {

            joint for joint in ("hip", "knee", "ankle")

            if re.search(r"\b" + joint + r"\b", answer_lower)

        }



        unsupported_joints = mentioned_joints - available_joints



        # Reject decimal numbers not present in the supplied

        # evidence, question, or authoritative classification.

        allowed_text = evidence + "\n" + question + "\n" + classification_facts



        allowed_numbers = [
            float(value)
            for value in re.findall(r"\b\d+\.\d+\b", allowed_text)
        ]

        answer_numbers = [
            float(value)
            for value in re.findall(r"\b\d+\.\d+\b", answer)
        ]

        unsupported_numbers = []

        for value in answer_numbers:
            matched = any(
                abs(value - allowed) <= max(0.0001, abs(allowed) * 0.001)
                for allowed in allowed_numbers
            )

            if not matched:
                unsupported_numbers.append(value)



        if unsupported_joints or unsupported_numbers:

            print("\n[GROUNDING CHECK] Generated answer failed validation.")

            print("Unsupported joints:", sorted(unsupported_joints))

            print("Unsupported numerical values:", sorted(unsupported_numbers))

            print("[GROUNDING CHECK] Returning verified evidence instead.")



            verified_lines = [

                line.strip()

                for line in evidence.splitlines()

                if line.strip().startswith("temporal_")

                and ":" in line

            ]



            if verified_lines:

                fallback = (

                    "The generated explanation contained information "

                    "that could not be verified against the retrieved "

                    "evidence. Here are the verified measurements "

                    "available for this recording:\n\n"

                )



                fallback += "\n".join(

                    f"- {line}" for line in verified_lines

                )



                fallback += (

                    "\n\nThese are image-coordinate-derived features, "

                    "not calibrated physical or clinically validated "

                    "measurements."

                )



                return fallback



            return (

                "The generated explanation could not be verified "

                "against the retrieved evidence. No unsupported "

                "measurements are being reported."

            )



        return answer





    except urllib.error.URLError as error:



        return (

            "Could not connect to Ollama. "

            "Check that Ollama is running. "

            f"Details: {error}"

        )



    except Exception as error:



        print(f"\\n[LLM FALLBACK] Generation failed: {error}")
        print("[LLM FALLBACK] Creating report from verified video evidence.")

        clean_evidence = evidence.replace("\\\\n", "\\n")
        clean_classification = classification_facts.replace("\\\\n", "\\n")

        probabilities = {}
        for line in clean_evidence.splitlines():
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            label = label.strip().lower()
            if label in ("normal probability", "abnormal probability"):
                probabilities[label] = value.strip()

        selected_features = (
            "left_hip_angle",
            "right_hip_angle",
            "left_knee_angle",
            "right_knee_angle"
        )

        display_names = {
            "left_hip_angle": "Left hip",
            "right_hip_angle": "Right hip",
            "left_knee_angle": "Left knee",
            "right_knee_angle": "Right knee"
        }

        verified_stats = {}
        for line in clean_evidence.splitlines():
            if ":" not in line:
                continue
            name, raw_stats = line.split(":", 1)
            name = name.strip()
            if name not in selected_features:
                continue
            try:
                stats = json.loads(raw_stats.strip())
                if "mean" in stats and "std" in stats:
                    verified_stats[name] = stats
            except (ValueError, TypeError):
                continue

        fallback_lines = [
            "GAITXPLAIN VIDEO ANALYSIS",
            "",
            "CLASSIFICATION",
            clean_classification.strip(),
            ""
        ]

        for label in ("normal probability", "abnormal probability"):
            if label in probabilities:
                try:
                    value = float(probabilities[label])
                    fallback_lines.append(
                        f"{label.title()}: {value:.4%}"
                    )
                except ValueError:
                    fallback_lines.append(
                        f"{label.title()}: {probabilities[label]}"
                    )

        fallback_lines.extend(["", "JOINT ANGLE SUMMARY"])

        for name in selected_features:
            stats = verified_stats.get(name)
            if stats is not None:
                fallback_lines.append(
                    f"{display_names[name]}: "
                    f"{stats['mean']:.2f} degrees "
                    f"(SD {stats['std']:.2f})"
                )

        if not verified_stats:
            fallback_lines.append(
                "Joint-angle statistics were not available."
            )

        # Three-layer biomechanical interpretation
        hip_l = verified_stats.get("left_hip_angle")
        hip_r = verified_stats.get("right_hip_angle")
        knee_l = verified_stats.get("left_knee_angle")
        knee_r = verified_stats.get("right_knee_angle")

        interpretation_lines = [
            "",
            "LAYER 1 - OBSERVATIONS",
        ]

        if hip_l and hip_r:
            hip_diff = abs(float(hip_l["mean"]) - float(hip_r["mean"]))
            interpretation_lines.append(
                f"The mean left-right hip angle difference is "
                f"{hip_diff:.2f} degrees."
            )

        if knee_l and knee_r:
            knee_diff = abs(float(knee_l["mean"]) - float(knee_r["mean"]))
            interpretation_lines.append(
                f"The mean left-right knee angle difference is "
                f"{knee_diff:.2f} degrees."
            )

        if not (hip_l and hip_r and knee_l and knee_r):
            interpretation_lines.append(
                "Some bilateral joint-angle measurements were unavailable."
            )

        interpretation_lines.extend([
            "",
            "LAYER 2 - BIOMECHANICAL INTERPRETATION",
        ])

        variability = []
        for label, left, right in (
            ("hip", hip_l, hip_r),
            ("knee", knee_l, knee_r)
        ):
            if left and right:
                left_sd = float(left["std"])
                right_sd = float(right["std"])
                variability.append((label, left_sd, right_sd))

        if variability:
            for label, left_sd, right_sd in variability:
                if abs(left_sd - right_sd) < 0.01:
                    description = "similar measured variability on both sides"
                elif left_sd > right_sd:
                    description = "greater measured variability on the left side"
                else:
                    description = "greater measured variability on the right side"

                interpretation_lines.append(
                    f"The {label} angle standard deviations indicate "
                    f"{description} across the analysed frames "
                    f"(left SD: {left_sd:.2f}, right SD: {right_sd:.2f})."
                )

            interpretation_lines.append(
                "These standard deviations describe variation in the "
                "extracted angle measurements; they do not by themselves "
                "establish irregular or impaired walking."
            )

        interpretation_lines.extend([
            "",
            "LAYER 3 - CLASSIFICATION RELEVANCE",
            "The model's classification is based on the combined "
            "32-feature representation and temporal information. "
            "The joint-angle observations provide additional descriptive "
            "context, but no individual angle or bilateral difference "
            "independently proves the predicted gait category.",
            "",
            "REFERENCE AND LIMITATION",
            "No validated reference range matched for camera setup, "
            "walking conditions and measurement method was supplied. "
            "Therefore, these values cannot be labelled higher or lower "
            "than an expected clinical range."
        ])

        fallback_lines.extend(interpretation_lines)
        fallback_lines.extend([
            "",
            "GENERAL LIMITATIONS",
            "Measurements are image-coordinate-derived, not calibrated "
            "physical measurements. The model was trained on a small "
            "dataset and has not established clinical validity or "
            "independent-subject generalization. This output is not "
            "a medical diagnosis.",
            "",
            "LLM STATUS",
            "The language model was unavailable or timed out. This report "
            "was generated from verified prediction and feature evidence."
        ])

        return "\\n".join(fallback_lines)





# ============================================================

# INTERACTIVE ASSISTANT

# ============================================================



def main():



    print("\n" + "=" * 65)

    print("              GAITXPLAIN AI ASSISTANT")

    print("=" * 65)



    print("LLM:", LLM_MODEL)

    print("Embedding device:", DEVICE)

    print("RAG collection:", COLLECTION_NAME)



    print("\nAsk questions about the indexed recordings.")

    print("Type 'exit' to close.")



    while True:



        print("\n" + "-" * 65)



        question = input("\nAsk GaitXplain: ").strip()



        if question.lower() in ["exit", "quit"]:

            print("\nGaitXplain session ended.")

            break



        if not question:

            continue



        recording_id = input(

            "Recording ID (Enter for all): "

        ).strip()



        recording_id = recording_id or None



        print("\nRetrieving evidence...")



        try:



            evidence, sources, feature_group, prediction = (

                retrieve_evidence(question, recording_id)

            )



        except Exception as error:



            print("Retrieval failed:", error)

            continue



        if not sources:



            print("No matching evidence found.")

            continue



        if feature_group:

            print("Feature routing:", feature_group)



        if prediction:



            metadata = prediction["metadata"]



            print(

                "Verified final classification:",

                determine_final_classification(metadata)

            )



        print("Retrieved evidence documents:", len(sources))

        print("\nGenerating explanation with Qwen...")



        answer = generate_explanation(

            question,

            evidence,

            recording_id,

            prediction

        )



        print("\n" + "=" * 65)

        print("              GAITXPLAIN RESPONSE")

        print("=" * 65)



        print("\n" + answer)



        print("\n" + "-" * 65)

        print("RETRIEVED SOURCES")



        for source in sources:



            print(

                source["recording_id"],

                "|",

                source["chunk_type"],

                "| Distance:",

                source["cosine_distance"]

            )



        print("=" * 65)





if __name__ == "__main__":

    main()



