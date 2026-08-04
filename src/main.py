import os
import pandas as pd
from data_loader import load_data, inspect_data_structure
from data_preprocessor import flatten_data, check_data_integrity
from eda import perform_eda
from data_visualizer import visualize_embeddings_tsne
from knn_classifier import prepare_data_for_classification, CustomKNeighborsClassifier
from model_evaluator import ModelEvaluator
from results_visualizer import ResultsVisualizer
from sklearn.model_selection import train_test_split

def main():
    print("\n==============================================")
    print(" Iniciando Pipeline ML Practical Test")
    print("==============================================")
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    
    raw_data = load_data(DATA_FILE_PATH)
    inspect_data_structure(raw_data)
    
    processed_df = flatten_data(raw_data)
    check_data_integrity(processed_df)
    perform_eda(processed_df)
    
    visualize_embeddings_tsne(processed_df)
    
    X, y, unique_syndromes = prepare_data_for_classification(processed_df)
    
    evaluator = ModelEvaluator(X, y, unique_syndromes)
    k_values_to_test = list(range(1, 16))
    
    euclidean_results_df = evaluator.evaluate_knn_with_cross_validation(k_values_to_test, metric='euclidean', n_splits=10)
    optimal_k_euclidean = evaluator.find_optimal_k(euclidean_results_df, 'f1_mean')
    print(f"Melhor k (Euclidiana, otimizando F1-Score): {optimal_k_euclidean}")
    
    cosine_results_df = evaluator.evaluate_knn_with_cross_validation(k_values_to_test, metric='cosine', n_splits=10)
    optimal_k_cosine = evaluator.find_optimal_k(cosine_results_df, 'f1_mean')
    print(f"Melhor k (Cosseno, otimizando F1-Score): {optimal_k_cosine}")
    
    results_visualizer = ResultsVisualizer()
    
    X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_euclidean_optimal = CustomKNeighborsClassifier(n_neighbors=optimal_k_euclidean, metric='euclidean')
    knn_euclidean_optimal.fit(X_train_roc, y_train_roc)
    y_proba_euclidean_roc = knn_euclidean_optimal.predict_proba(X_test_roc)
    
    knn_cosine_optimal = CustomKNeighborsClassifier(n_neighbors=optimal_k_cosine, metric='cosine')
    knn_cosine_optimal.fit(X_train_roc, y_train_roc)
    y_proba_cosine_roc = knn_cosine_optimal.predict_proba(X_test_roc)
    
    results_visualizer.plot_roc_curves([y_test_roc], [y_proba_euclidean_roc], unique_syndromes, 'euclidean')
    results_visualizer.plot_roc_curves([y_test_roc], [y_proba_cosine_roc], unique_syndromes, 'cosine')
    
    results_visualizer.plot_k_optimization_results(euclidean_results_df, cosine_results_df)
    results_visualizer.generate_summary_tables(euclidean_results_df, cosine_results_df, optimal_k_euclidean, optimal_k_cosine)
    
    print("\n==============================================")
    print(" Pipeline ML Practical Test")
    print("==============================================")
    return processed_df, X, y, unique_syndromes, euclidean_results_df, cosine_results_df

if __name__ == "__main__":
    final_df, X_data, y_labels, syndromes, euclidean_res, cosine_res = main()