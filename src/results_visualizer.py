import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc

class ResultsVisualizer:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_roc_curves(self, y_true_folds, y_proba_folds, unique_syndromes, metric_name):
        print(f"\n - Gerando Curvas ROC para {metric_name} -")
        plt.figure(figsize=(10, 8))
        n_classes = len(unique_syndromes)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for i in range(n_classes):
            fprs_class = []
            tprs_class = []
            aucs_class = []
            
            for fold_idx in range(len(y_true_folds)):
                y_true_binary = (y_true_folds[fold_idx] == i).astype(int)
                y_proba_binary = y_proba_folds[fold_idx][:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
                roc_auc = auc(fpr, tpr)
                
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_class.append(interp_tpr)
                aucs_class.append(roc_auc)
                
            mean_tpr_class = np.mean(tprs_class, axis=0)
            mean_tpr_class[-1] = 1.0
            mean_auc_class = auc(mean_fpr, mean_tpr_class)
            
            plt.plot(mean_fpr, mean_tpr_class, label=f'Classe {unique_syndromes[i]} (AUC = {mean_auc_class:.2f})')
            tprs.append(mean_tpr_class)
            aucs.append(mean_auc_class)
            
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Acaso')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC Média (OvR) para KNN com Distância {metric_name.capitalize()}')
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(self.output_dir, f'roc_curve_knn_{metric_name}.png')
        plt.savefig(plot_path)
        print(f"Curva ROC para {metric_name} salva em: {plot_path}")
        plt.close()

    def plot_k_optimization_results(self, euclidean_results_df, cosine_results_df):
        print("\n - Gerando Gráficos de Otimização de k -")
        metrics = ['accuracy_mean', 'f1_mean', 'auc_mean', 'top3_acc_mean']
        metric_titles = {
            'accuracy_mean': 'Acurácia Média',
            'f1_mean': 'F1-Score Médio',
            'auc_mean': 'AUC Médio (OvR)',
            'top3_acc_mean': 'Top-3 Acurácia Média'
        }
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(euclidean_results_df['k'], euclidean_results_df[metric], marker='o', label='Euclidiana')
            plt.plot(cosine_results_df['k'], cosine_results_df[metric], marker='s', label='Cosseno')
            plt.title(f'Desempenho do KNN vs. k ({metric_titles[metric]})')
            plt.xlabel('Valor de k')
            plt.ylabel(metric_titles[metric])
            plt.xticks(euclidean_results_df['k'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plot_path = os.path.join(self.output_dir, f'k_optimization_{metric}.png')
            plt.savefig(plot_path)
            print(f"Gráfico de otimização de k para {metric_titles[metric]} salvo em: {plot_path}")
            plt.close()

    def generate_summary_tables(self, euclidean_results_df, cosine_results_df, optimal_k_euclidean, optimal_k_cosine):
        print("\n - Gerando Tabelas de Resumo de Desempenho -")
        
        optimal_euclidean_row = euclidean_results_df[euclidean_results_df['k'] == optimal_k_euclidean].iloc[0]
        optimal_cosine_row = cosine_results_df[cosine_results_df['k'] == optimal_k_cosine].iloc[0]
        
        summary_data = {
            'Métrica': ['Acurácia', 'F1-Score', 'AUC (OvR)', 'Top-3 Acurácia'],
            f'Euclidiana (k={optimal_k_euclidean})': [
                f"{optimal_euclidean_row['accuracy_mean']:.4f} ± {optimal_euclidean_row['accuracy_std']:.4f}",
                f"{optimal_euclidean_row['f1_mean']:.4f} ± {optimal_euclidean_row['f1_std']:.4f}",
                f"{optimal_euclidean_row['auc_mean']:.4f} ± {optimal_euclidean_row['auc_std']:.4f}",
                f"{optimal_euclidean_row['top3_acc_mean']:.4f} ± {optimal_euclidean_row['top3_acc_std']:.4f}"
            ],
            f'Cosseno (k={optimal_k_cosine})': [
                f"{optimal_cosine_row['accuracy_mean']:.4f} ± {optimal_cosine_row['accuracy_std']:.4f}",
                f"{optimal_cosine_row['f1_mean']:.4f} ± {optimal_cosine_row['f1_std']:.4f}",
                f"{optimal_cosine_row['auc_mean']:.4f} ± {optimal_cosine_row['auc_std']:.4f}",
                f"{optimal_cosine_row['top3_acc_mean']:.4f} ± {optimal_cosine_row['top3_acc_std']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n - Tabela de Resumo de Desempenho -")
        print(summary_df.to_markdown(index=False))
        
        table_path = os.path.join(self.output_dir, 'performance_summary_table.md')
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(summary_df.to_markdown(index=False))
            
        print(f"Tabela de resumo salva em: {table_path}")

if __name__ == "__main__":
    print("Este script é projetado para ser usado como um módulo. Execute o arquivo main.py.")