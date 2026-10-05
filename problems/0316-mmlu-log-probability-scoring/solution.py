import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Converts log-probabilities/logits into probabilities.
    """

    # Subtract max for numerical stability
    shifted = logits - np.max(logits)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


def mmlu_log_prob_score(
    log_probs: list,
    correct_answers: list
) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.

    Args:
        log_probs:
            List of lists where each inner list contains
            log-probabilities for answer choices.

        correct_answers:
            List of correct answer indices (0-indexed)

    Returns:
        Dictionary containing:
            - accuracy
            - predictions
            - avg_correct_prob
    """

    # -----------------------------------------
    # Validation
    # -----------------------------------------
    if len(log_probs) != len(correct_answers):
        raise ValueError(
            "log_probs and correct_answers "
            "must have same length."
        )

    predictions = []

    total_correct = 0

    correct_probs = []

    # -----------------------------------------
    # Process each question
    # -----------------------------------------
    for probs, correct_idx in zip(
        log_probs,
        correct_answers
    ):

        probs = np.array(probs, dtype=float)

        # -------------------------------------
        # Prediction = argmax(log-probability)
        # -------------------------------------
        predicted_idx = int(np.argmax(probs))

        predictions.append(predicted_idx)

        # -------------------------------------
        # Accuracy tracking
        # -------------------------------------
        if predicted_idx == correct_idx:
            total_correct += 1

        # -------------------------------------
        # Convert to probabilities
        # using stable softmax
        # -------------------------------------
        probabilities = stable_softmax(probs)

        # Probability assigned to correct answer
        correct_prob = probabilities[correct_idx]

        correct_probs.append(correct_prob)

    # -----------------------------------------
    # Final metrics
    # -----------------------------------------
    accuracy = (
        total_correct / len(log_probs)
        if len(log_probs) > 0
        else 0.0
    )

    avg_correct_prob = (
        float(np.mean(correct_probs))
        if correct_probs
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "avg_correct_prob": avg_correct_prob
    }


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
# if __name__ == "__main__":

#     log_probs = [
#         [-2.1, -0.3, -1.5, -3.0],
#         [-1.2, -2.5, -0.1, -4.0],
#         [-0.2, -1.8, -3.1, -2.2]
#     ]

#     correct_answers = [1, 2, 0]

#     results = mmlu_log_prob_score(
#         log_probs,
#         correct_answers
#     )

#     print(results)