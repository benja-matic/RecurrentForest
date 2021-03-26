import numpy as np
from sklearn import metrics

def binary_metrics(y, y_hat, verbose=True, model=None):

    acc = np.mean(y_hat == y)
    precision = metrics.precision_score(y, y_hat)
    recall = metrics.recall_score(y, y_hat)
    f1 = metrics.f1_score(y, y_hat)

    if verbose:
        metric_names = ["Accuracy", "Precision" "Recall", "F1"]
        metric_vals = [acc, precision, recall, f1]
        if model:
            print(f"Metrics for {model}\n\n")
        for i in range(len(metric_names)):
            print(f"{metric_names[i]}: {metric_vals[i]}")

    return acc, precision, recall, f1
