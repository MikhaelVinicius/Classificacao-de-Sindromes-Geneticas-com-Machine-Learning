import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from knn_classifier import CustomKNeighborsClassifier
from metrics_calculator import MetricsCalculator

class ModelEvaluator:
    def __init__(self, X, y, unique_syndromes):
        self.X = X
        self.y = y
        self.unique_syndromes = unique_syndromes
        self.metrics_calculator = MetricsCalculator()

    def evaluate_knn_with_cross_validation(self, k_values, metric, n_splits=10):
        print(f"\n - Iniciando Validação Cruzada ({n_splits}-Fold) para métrica: {metric} - ")
        results = {
            "k": [],
            "accuracy_mean": [], "accuracy_std": [],
            "f1_mean": [], "f1_std": [],
            "auc_mean": [], "auc_std": [],
            "top3_acc_mean": [], "top3_acc_std": []
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for k in k_values:
            print(f"Testando k = {k} .")
            fold_accuracies = []
            fold_f1_scores = []
            fold_auc_scores = []
            fold_top3_accuracies = []
            
            for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                
                knn = CustomKNeighborsClassifier(n_neighbors=k, metric=metric)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                y_proba = knn.predict_proba(X_test)
                
                fold_accuracies.append(accuracy_score(y_test, y_pred))
                fold_f1_scores.append(self.metrics_calculator.calculate_f1_score(y_test, y_pred))
                fold_top3_accuracies.append(self.metrics_calculator.calculate_top_k_accuracy(y_test, y_proba, k=3))
                
                auc_per_class = []
                for class_idx in range(len(self.unique_syndromes)):
                    _, _, _, auc_score = self.metrics_calculator.calculate_roc_auc(y_test, y_proba, class_idx)
                    auc_per_class.append(auc_score)
                fold_auc_scores.append(np.mean(auc_per_class))
                
            results["k"].append(k)
            results["accuracy_mean"].append(np.mean(fold_accuracies))
            results["accuracy_std"].append(np.std(fold_accuracies))
            results["f1_mean"].append(np.mean(fold_f1_scores))
            results["f1_std"].append(np.std(fold_f1_scores))
            results["auc_mean"].append(np.mean(fold_auc_scores))
            results["auc_std"].append(np.std(fold_auc_scores))
            results["top3_acc_mean"].append(np.mean(fold_top3_accuracies))
            results["top3_acc_std"].append(np.std(fold_top3_accuracies))
            
        print(f"Validação Cruzada para métrica {metric.capitalize()} concluída.")
        return pd.DataFrame(results)

    def find_optimal_k(self, results_df, metric_to_optimize='accuracy_mean'):
        optimal_k = results_df.loc[results_df[metric_to_optimize].idxmax()]['k']
        print(f"Valor ótimo de k para {metric_to_optimize}: {int(optimal_k)}")
        return int(optimal_k)

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessor import flatten_data
    from knn_classifier import prepare_data_for_classification
    import os
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    processed_df = flatten_data(raw_data)
    X, y, unique_syndromes = prepare_data_for_classification(processed_df)
    
    evaluator = ModelEvaluator(X, y, unique_syndromes)
    k_values_to_test = list(range(1, 16))
    
    euclidean_results = evaluator.evaluate_knn_with_cross_validation(k_values_to_test, metric='euclidean', n_splits=10)
    print("\nResultados Euclidiana:\n", euclidean_results)
    optimal_k_euclidean = evaluator.find_optimal_k(euclidean_results, 'f1_mean')
    print(f"Melhor k (Euclidiana, otimizando F1-Score): {optimal_k_euclidean}")
    
    cosine_results = evaluator.evaluate_knn_with_cross_validation(k_values_to_test, metric='cosine', n_splits=10)
    print("\nResultados Cosseno:\n", cosine_results)
    optimal_k_cosine = evaluator.find_optimal_k(cosine_results, 'f1_mean')
    print(f"Melhor k (Cosseno, otimizando F1-Score): {optimal_k_cosine}")