import re
from collections import defaultdict

def extract_answer_letter(text: str) -> str | None:
    """
    Extract answer letter (A/B/C/D) from model output.

    Handles formats like:
    - "A"
    - "a"
    - "A."
    - "(A)"
    - "A)"
    - "The answer is B"
    - "I think it's C because..."
    """

    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # -----------------------------------------
    # Pattern 1:
    # Standalone answer formats
    # -----------------------------------------
    patterns = [
        r'^\(?\s*([ABCD])\s*[\.\)]?\s*$',      # A / A. / (A)
        r'answer\s*(is|:)?\s*\(?([ABCD])\)?', # answer is B
        r'option\s*\(?([ABCD])\)?',           # option C
        r'choose\s*\(?([ABCD])\)?',           # choose D
        r'\b([ABCD])\b'                       # standalone letter
    ]

    upper_text = text.upper()

    for pattern in patterns:

        match = re.search(pattern, upper_text)

        if match:

            # Extract last non-None group
            groups = match.groups()

            for g in reversed(groups):
                if g in {"A", "B", "C", "D"}:
                    return g

    return None


def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics
    """

    # -----------------------------------------
    # Validation
    # -----------------------------------------
    if not (
        len(model_outputs)
        == len(ground_truth)
        == len(subjects)
    ):
        raise ValueError(
            "All input lists must have same length."
        )

    total_questions = len(model_outputs)

    total_correct = 0
    valid_responses = 0

    subject_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    # -----------------------------------------
    # Evaluate predictions
    # -----------------------------------------
    for output, truth, subject in zip(
        model_outputs,
        ground_truth,
        subjects
    ):

        truth = truth.upper()

        predicted = extract_answer_letter(output)

        subject_stats[subject]["total"] += 1

        if predicted is not None:
            valid_responses += 1

        if predicted == truth:
            total_correct += 1
            subject_stats[subject]["correct"] += 1

    # -----------------------------------------
    # Compute subject-wise accuracy
    # -----------------------------------------
    subject_accuracy = {}

    for subject, stats in subject_stats.items():

        accuracy = (
            stats["correct"] / stats["total"]
            if stats["total"] > 0
            else 0.0
        )

        subject_accuracy[subject] = accuracy

    # -----------------------------------------
    # Overall metrics
    # -----------------------------------------
    overall_accuracy = (
        total_correct / total_questions
        if total_questions > 0
        else 0.0
    )

    valid_response_rate = (
        valid_responses / total_questions
        if total_questions > 0
        else 0.0
    )

    return {
        "overall_accuracy": overall_accuracy,
        "subject_accuracy": subject_accuracy,
        "valid_response_rate": valid_response_rate,
        "total_correct": total_correct,
        "total_questions": total_questions
    }