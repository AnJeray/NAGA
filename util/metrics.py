# metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def calculate_metrics(y_true: np.ndarray, y_pre: np.ndarray) -> tuple:

    accuracy = accuracy_score(y_true, y_pre)
    f1_w = f1_score(y_true, y_pre, average='weighted')
    recall_w = recall_score(y_true, y_pre, average='weighted')
    precision_w = precision_score(y_true, y_pre, average='weighted')
    f1 = f1_score(y_true, y_pre, average='macro')
    recall = recall_score(y_true, y_pre, average='macro')
    precision = precision_score(y_true, y_pre, average='macro')
    return accuracy, f1_w, recall_w, precision_w, f1, recall, precision
