import csv
import os
import sys

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
# VERDICT NORMALIZATION
# ============================================================

def normalize_verdict(verdict):

    verdict = str(verdict).upper().strip()

    if verdict in ["REAL", "MOSTLY REAL"]:
        return "REAL"

    if verdict in ["FAKE", "MOSTLY FALSE"]:
        return "FAKE"

    if verdict in ["MISLEADING", "PARTIALLY TRUE"]:
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
# EVALUATION
# ============================================================

def evaluate():

    rows = load_dataset()

    if not rows:
        print("\nERROR: Dataset contains no rows.")
        return

    results = []

    print("\n======================================")
    print(" AI FACT CHECKER EVALUATION")
    print("======================================\n")

    # --------------------------------------------------------
    # RUN EACH CLAIM
    # --------------------------------------------------------

    for index, row in enumerate(rows, start=1):

        claim = row["claim"].strip()

        expected = normalize_verdict(
            row["ground_truth"]
        )

        print(
            f"[{index}/{len(rows)}] Checking:"
        )

        print(claim)

        try:

            # Call Gemini fact checker
            result = analyze_news_text(claim)

            # Extract prediction
            predicted_raw = result.get(
                "verdict",
                "UNVERIFIED"
            )

            predicted = normalize_verdict(
                predicted_raw
            )

            # Extract confidence
            confidence = result.get(
                "confidence",
                0
            )

            # Check correctness
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

        except Exception as e:

            print(
                f"ERROR      : {e}"
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
    # FILTER SUCCESSFUL RESULTS
    # ========================================================

    valid_results = [
        result
        for result in results
        if result["predicted"] != "ERROR"
    ]

    if not valid_results:

        print(
            "\nNo valid predictions were produced."
        )

        return


    # ========================================================
    # TRUE AND PREDICTED LABELS
    # ========================================================

    y_true = [
        result["ground_truth"]
        for result in valid_results
    ]

    y_pred = [
        result["predicted"]
        for result in valid_results
    ]


    # ========================================================
    # LABELS
    # ========================================================

    labels = [
        "REAL",
        "FAKE",
        "MISLEADING",
        "UNVERIFIED"
    ]


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


    # ========================================================
    # RESULTS
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
    # PER-CLASS METRICS
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

        print(
            f"\n{label}"
        )

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
    # SAVE EVALUATION RESULTS
    # ========================================================

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


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "Evaluation completed successfully."
    )

    print(
        f"Results saved to:\n{results_path}"
    )

    print(
        "======================================\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()