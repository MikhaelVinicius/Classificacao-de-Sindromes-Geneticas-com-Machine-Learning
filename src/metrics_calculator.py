import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix

class MetricsCalculator:
    def __init__(self):
        pass

    def calculate_roc_auc(self, y_true, y_proba, class_index):
        y_true_binary = (y_true == class_index).astype(int)
        
        if y_proba.ndim > 1:
            y_proba_binary = y_proba[:, class_index]
        else:
            y_proba_binary = y_proba
            
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba_binary)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_score

    def calculate_f1_score(self, y_true, y_pred, average='weighted'):
        return f1_score(y_true, y_pred, average=average)

    def calculate_top_k_accuracy(self, y_true, y_proba, k=1):
        num_samples = y_true.shape[0]
        correct_predictions = 0
        
        for i in range(num_samples):
            top_k_indices = np.argsort(y_proba[i])[-k:]
            if y_true[i] in top_k_indices:
                correct_predictions += 1
                
        return correct_predictions / num_samples

if __name__ == "__main__":
    y_true_example = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_proba_example = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    y_pred_example = np.argmax(y_proba_example, axis=1)
    metrics_calc = MetricsCalculator()
    
    print("\n - Testando F1-Score -")
    f1 = metrics_calc.calculate_f1_score(y_true_example, y_pred_example, average='weighted')
    print(f"F1-Score (weighted): {f1:.4f}")
    
    print("\n - Testando Top-k Accuracy -")
    top1_acc = metrics_calc.calculate_top_k_accuracy(y_true_example, y_proba_example, k=1)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    
    top2_acc = metrics_calc.calculate_top_k_accuracy(y_true_example, y_proba_example, k=2)
    print(f"Top-2 Accuracy: {top2_acc:.4f}")
    
    top3_acc = metrics_calc.calculate_top_k_accuracy(y_true_example, y_proba_example, k=3)
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    print("\n - Testando ROC AUC (para classe 0) -")
    fpr, tpr, thresholds, auc_score = metrics_calc.calculate_roc_auc(y_true_example, y_proba_example, class_index=0)
    print(f"AUC para classe 0: {auc_score:.4f}")