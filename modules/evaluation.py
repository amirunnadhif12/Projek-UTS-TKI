"""
Evaluation module — Mini Search Engine
========================================
Menghitung Precision, Recall, dan F1-Score.
"""


def evaluate(predicted: list[int], ground_truth: list[int]) -> dict:
    """
    Hitung metrik evaluasi.

    Parameters
    ----------
    predicted    : list[int] — indeks dokumen yang di-retrieve sistem
    ground_truth : list[int] — indeks dokumen yang benar-benar relevan

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1
    """
    pred_set = set(predicted)
    truth_set = set(ground_truth)

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }