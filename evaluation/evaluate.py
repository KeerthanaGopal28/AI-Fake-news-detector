import csv
import os
import sys
import time
import json
import hashlib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from fact_checker import analyze_news_text


# ============================================================
# CACHE PATH
# ============================================================

CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "cache"
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# CACHE FUNCTIONS
# ============================================================

def get_cache_file(claim):

    claim_hash = hashlib.sha256(
        claim.strip().encode("utf-8")
    ).hexdigest()

    return os.path.join(
        CACHE_DIR,
        f"{claim_hash}.json"
    )


def get_cached_result(claim):

    cache_file = get_cache_file(claim)

    if not os.path.exists(cache_file):
        return None

    try:

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return None


# ============================================================
# VERDICT NORMALIZATION
# ============================================================

def normalize_verdict(verdict):

    verdict = str(verdict).upper().strip()

    if verdict in [
        "REAL",
        "MOSTLY REAL"
    ]:
        return "REAL"

    if verdict in [
        "FAKE",
        "MOSTLY FALSE"
    ]:
        return "FAKE"

    if verdict in [
        "MISLEADING",
        "PARTIALLY TRUE"
    ]:
        return "MISLEADING"

    return "UNVERIFIED"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    dataset_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "fact_check_dataset.csv"
    )

    print("Dataset path:")
    print(dataset_path)

    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with open(
        dataset_path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print(
        f"\nDataset rows loaded: {len(rows)}"
    )

    return rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    results_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "evaluation_results.csv"
    )

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "id",
            "claim",
            "ground_truth",
            "predicted",
            "confidence",
            "correct"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    return results_path


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    rows = load_dataset()

    results = []

    print("\n======================================")
    print(" AI FACT CHECKER EVALUATION")
    print("======================================\n")


    # ========================================================
    # PROCESS CLAIMS
    # ========================================================

    for index, row in enumerate(rows, start=1):

        claim = row["claim"].strip()

        expected = normalize_verdict(
            row["ground_truth"]
        )

        print(
            f"[{index}/{len(rows)}] Checking:"
        )

        print(claim)


        # ====================================================
        # CHECK CACHE FIRST
        # ====================================================

        cached = get_cached_result(claim)

        if cached is not None:

            print(
                "Using cached result. ✓"
            )

            predicted = normalize_verdict(
                cached.get(
                    "verdict",
                    "UNVERIFIED"
                )
            )

            confidence = cached.get(
                "confidence",
                0
            )

            correct = (
                expected == predicted
            )

            results.append({
                "id": row["id"],
                "claim": claim,
                "ground_truth": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": correct
            })

            print(
                f"Expected   : {expected}"
            )

            print(
                f"Predicted  : {predicted}"
            )

            print(
                f"Confidence : {confidence}%"
            )

            if correct:

                print(
                    "Result     : ✓ CORRECT"
                )

            else:

                print(
                    "Result     : ✗ WRONG"
                )

            print("-" * 60)

            continue


        # ====================================================
        # API CALL FOR MISSING CLAIM
        # ====================================================

        print(
            "No cache found. Calling Gemini..."
        )

        try:

            result = analyze_news_text(
                claim
            )

            predicted = normalize_verdict(
                result.get(
                    "verdict",
                    "UNVERIFIED"
                )
            )

            confidence = result.get(
                "confidence",
                0
            )

            correct = (
                expected == predicted
            )

            results.append({
                "id": row["id"],
                "claim": claim,
                "ground_truth": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": correct
            })

            print(
                f"Expected   : {expected}"
            )

            print(
                f"Predicted  : {predicted}"
            )

            print(
                f"Confidence : {confidence}%"
            )

            if correct:

                print(
                    "Result     : ✓ CORRECT"
                )

            else:

                print(
                    "Result     : ✗ WRONG"
                )


            # ------------------------------------------------
            # WAIT BEFORE NEXT API REQUEST
            # ------------------------------------------------

            print(
                "Waiting 20 seconds before next API request..."
            )

            time.sleep(20)


        except Exception as e:

            print(
                f"ERROR      : {e}"
            )

            print(
                "Skipping this claim for now."
            )

            results.append({
                "id": row["id"],
                "claim": claim,
                "ground_truth": expected,
                "predicted": "ERROR",
                "confidence": 0,
                "correct": False
            })


        print("-" * 60)


    # ========================================================
    # VALID RESULTS
    # ========================================================

    valid_results = [
        result
        for result in results
        if result["predicted"] != "ERROR"
    ]


    if not valid_results:

        print(
            "\nNo valid predictions available."
        )

        return


    # ========================================================
    # METRICS
    # ========================================================

    y_true = [
        result["ground_truth"]
        for result in valid_results
    ]

    y_pred = [
        result["predicted"]
        for result in valid_results
    ]


    labels = [
        "REAL",
        "FAKE",
        "MISLEADING",
        "UNVERIFIED"
    ]


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )
    )


    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\n======================================")
    print(" ML EVALUATION METRICS")
    print("======================================")

    print(
        f"Total Claims       : {len(results)}"
    )

    print(
        f"Valid Predictions  : {len(valid_results)}"
    )

    print(
        f"Failed Predictions : {len(results) - len(valid_results)}"
    )

    print(
        f"Accuracy           : {accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision : {precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Weighted F1-score  : {f1 * 100:.2f}%"
    )


    # ========================================================
    # PER CLASS METRICS
    # ========================================================

    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )


    print("\n======================================")
    print(" PER-CLASS METRICS")
    print("======================================")


    for i, label in enumerate(labels):

        print(f"\n{label}")

        print(
            f"  Precision : {class_precision[i] * 100:.2f}%"
        )

        print(
            f"  Recall    : {class_recall[i] * 100:.2f}%"
        )

        print(
            f"  F1-score  : {class_f1[i] * 100:.2f}%"
        )

        print(
            f"  Samples   : {class_support[i]}"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n======================================")
    print(" CONFUSION MATRIX")
    print("======================================")

    print(
        "\nLabels:"
    )

    print(labels)

    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted\n"
    )

    print(matrix)


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = save_results(
        results
    )


    print("\n======================================")
    print(
        "Evaluation completed."
    )

    print(
        f"Results saved to:\n{results_path}"
    )

    print(
        "======================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()