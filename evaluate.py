# evaluate.py
import torch
import numpy as np
from utils.metrics import dice_score


def compute_confusion_metrics(pred, target, num_classes, eps=1e-6):
    """
    Computes TP, FP, TN, FN per class
    """
    metrics = {}

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()
        tn = (~pred_c & ~target_c).sum().item()

        metrics[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        }

    return metrics


def compute_scores(confusion, eps=1e-6):
    """
    Computes Sensitivity, Specificity, Precision, Recall
    """
    scores = {}

    for c, m in confusion.items():
        tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]

        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        precision = tp / (tp + fp + eps)
        recall = sensitivity

        scores[c] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Recall": recall
        }

    return scores


def evaluate(model, loader, device, num_classes):
    """
    Full evaluation for medical image segmentation
    """
    model.eval()

    dice_scores = []
    all_confusion = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                     for c in range(num_classes)}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Dice
            dice_scores.append(dice_score(probs, y).item())

            # Confusion metrics
            batch_conf = compute_confusion_metrics(
                preds, y, num_classes)

            for c in range(num_classes):
                for k in all_confusion[c]:
                    all_confusion[c][k] += batch_conf[c][k]

    mean_dice = np.mean(dice_scores)
    class_scores = compute_scores(all_confusion)

    return mean_dice, class_scores


if __name__ == "__main__":
    print("Run evaluation through train.py or a dedicated evaluation script.")
