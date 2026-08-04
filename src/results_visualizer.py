import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from knn_classifier import CustomKNeighborsClassifier


class ResultsVisualizer:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_roc_curves(
        self,
        y_true_folds,
        y_proba_folds,
        unique_syndromes,
        metric_name
    ):
        print(f"\n - Gerando Curvas ROC para {metric_name} -")

        plt.figure(figsize=(10, 8))

        n_classes = len(unique_syndromes)
        mean_fpr = np.linspace(0, 1, 100)

        for class_idx in range(n_classes):
            class_tprs = []

            for fold_idx in range(len(y_true_folds)):
                y_true_binary = (
                    y_true_folds[fold_idx] == class_idx
                ).astype(int)

                y_proba_binary = y_proba_folds[fold_idx][:, class_idx]

                fpr, tpr, _ = roc_curve(
                    y_true_binary,
                    y_proba_binary
                )

                interpolated_tpr = np.interp(
                    mean_fpr,
                    fpr,
                    tpr
                )

                interpolated_tpr[0] = 0.0
                class_tprs.append(interpolated_tpr)

            mean_tpr_class = np.mean(class_tprs, axis=0)
            mean_tpr_class[-1] = 1.0

            mean_auc_class = auc(
                mean_fpr,
                mean_tpr_class
            )

            plt.plot(
                mean_fpr,
                mean_tpr_class,
                label=(
                    f"Classe {unique_syndromes[class_idx]} "
                    f"(AUC = {mean_auc_class:.2f})"
                )
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=2,
            color="red",
            label="Acaso"
        )

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "Curva ROC Média (OvR) para KNN com Distância "
            f"{metric_name.capitalize()}"
        )
        plt.legend(loc="lower right")
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            f"roc_curve_knn_{metric_name}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Curva ROC para {metric_name} salva em: "
            f"{plot_path}"
        )

    def plot_roc_comparison_cv(
        self,
        X,
        y,
        unique_syndromes,
        optimal_k_euclidean,
        optimal_k_cosine,
        n_splits=10
    ):
        print(
            "\n - Gerando Curva ROC Comparativa "
            "com Validação Cruzada -"
        )

        mean_fpr = np.linspace(0, 1, 200)

        configurations = {
            f"Euclidiana (k={optimal_k_euclidean})": {
                "metric": "euclidean",
                "k": optimal_k_euclidean
            },
            f"Cosseno (k={optimal_k_cosine})": {
                "metric": "cosine",
                "k": optimal_k_cosine
            }
        }

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        plt.figure(figsize=(10, 8))

        for model_name, config in configurations.items():
            fold_macro_tprs = []
            fold_macro_aucs = []

            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                knn = CustomKNeighborsClassifier(
                    n_neighbors=config["k"],
                    metric=config["metric"]
                )

                knn.fit(X_train, y_train)
                y_proba = knn.predict_proba(X_test)

                class_tprs = []

                for class_idx in range(len(unique_syndromes)):
                    y_true_binary = (
                        y_test == class_idx
                    ).astype(int)

                    fpr, tpr, _ = roc_curve(
                        y_true_binary,
                        y_proba[:, class_idx]
                    )

                    interpolated_tpr = np.interp(
                        mean_fpr,
                        fpr,
                        tpr
                    )

                    interpolated_tpr[0] = 0.0
                    class_tprs.append(interpolated_tpr)

                fold_macro_tpr = np.mean(
                    class_tprs,
                    axis=0
                )

                fold_macro_tpr[0] = 0.0
                fold_macro_tpr[-1] = 1.0

                fold_macro_tprs.append(fold_macro_tpr)

                fold_macro_aucs.append(
                    auc(mean_fpr, fold_macro_tpr)
                )

            mean_tpr = np.mean(
                fold_macro_tprs,
                axis=0
            )

            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0

            mean_auc = np.mean(fold_macro_aucs)
            std_auc = np.std(fold_macro_aucs)

            plt.plot(
                mean_fpr,
                mean_tpr,
                linewidth=2.5,
                label=(
                    f"{model_name} "
                    f"(AUC = {mean_auc:.4f} "
                    f"± {std_auc:.4f})"
                )
            )

            print(
                f"{model_name}: "
                f"AUC = {mean_auc:.4f} "
                f"± {std_auc:.4f}"
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            color="gray",
            label="Classificador aleatório"
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel("Taxa de Falsos Positivos")
        plt.ylabel("Taxa de Verdadeiros Positivos")
        plt.title(
            "Comparação das Curvas ROC Médias\n"
            "KNN Euclidiano vs. Cosseno"
        )
        plt.grid(
            True,
            linestyle="--",
            alpha=0.4
        )
        plt.legend(loc="lower right")
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            "roc_curve_comparison.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Curva ROC comparativa salva em: "
            f"{plot_path}"
        )

    def plot_k_optimization_results(
        self,
        euclidean_results_df,
        cosine_results_df
    ):
        print(
            "\n - Gerando Gráficos de Otimização de k -"
        )

        metrics = [
            "accuracy_mean",
            "f1_mean",
            "auc_mean",
            "top3_acc_mean"
        ]

        metric_titles = {
            "accuracy_mean": "Acurácia Média",
            "f1_mean": "F1-Score Médio",
            "auc_mean": "AUC Médio (OvR)",
            "top3_acc_mean": "Top-3 Acurácia Média"
        }

        for metric in metrics:
            plt.figure(figsize=(10, 6))

            plt.plot(
                euclidean_results_df["k"],
                euclidean_results_df[metric],
                marker="o",
                label="Euclidiana"
            )

            plt.plot(
                cosine_results_df["k"],
                cosine_results_df[metric],
                marker="s",
                label="Cosseno"
            )

            plt.title(
                "Desempenho do KNN vs. k "
                f"({metric_titles[metric]})"
            )
            plt.xlabel("Valor de k")
            plt.ylabel(metric_titles[metric])
            plt.xticks(euclidean_results_df["k"])
            plt.grid(
                True,
                linestyle="--",
                alpha=0.6
            )
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(
                self.output_dir,
                f"k_optimization_{metric}.png"
            )

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                "Gráfico de otimização de k para "
                f"{metric_titles[metric]} salvo em: "
                f"{plot_path}"
            )

    def generate_summary_tables(
        self,
        euclidean_results_df,
        cosine_results_df,
        optimal_k_euclidean,
        optimal_k_cosine
    ):
        print(
            "\n - Gerando Tabelas de Resumo "
            "de Desempenho -"
        )

        optimal_euclidean_row = euclidean_results_df[
            euclidean_results_df["k"]
            == optimal_k_euclidean
        ].iloc[0]

        optimal_cosine_row = cosine_results_df[
            cosine_results_df["k"]
            == optimal_k_cosine
        ].iloc[0]

        summary_data = {
            "Métrica": [
                "Acurácia",
                "F1-Score",
                "AUC (OvR)",
                "Top-3 Acurácia"
            ],
            f"Euclidiana (k={optimal_k_euclidean})": [
                (
                    f"{optimal_euclidean_row['accuracy_mean']:.4f} "
                    f"± {optimal_euclidean_row['accuracy_std']:.4f}"
                ),
                (
                    f"{optimal_euclidean_row['f1_mean']:.4f} "
                    f"± {optimal_euclidean_row['f1_std']:.4f}"
                ),
                (
                    f"{optimal_euclidean_row['auc_mean']:.4f} "
                    f"± {optimal_euclidean_row['auc_std']:.4f}"
                ),
                (
                    f"{optimal_euclidean_row['top3_acc_mean']:.4f} "
                    f"± {optimal_euclidean_row['top3_acc_std']:.4f}"
                )
            ],
            f"Cosseno (k={optimal_k_cosine})": [
                (
                    f"{optimal_cosine_row['accuracy_mean']:.4f} "
                    f"± {optimal_cosine_row['accuracy_std']:.4f}"
                ),
                (
                    f"{optimal_cosine_row['f1_mean']:.4f} "
                    f"± {optimal_cosine_row['f1_std']:.4f}"
                ),
                (
                    f"{optimal_cosine_row['auc_mean']:.4f} "
                    f"± {optimal_cosine_row['auc_std']:.4f}"
                ),
                (
                    f"{optimal_cosine_row['top3_acc_mean']:.4f} "
                    f"± {optimal_cosine_row['top3_acc_std']:.4f}"
                )
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        print(
            "\n - Tabela de Resumo de Desempenho -"
        )
        print(summary_df.to_markdown(index=False))

        table_path = os.path.join(
            self.output_dir,
            "performance_summary_table.md"
        )

        with open(
            table_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                summary_df.to_markdown(index=False)
            )

        print(
            f"Tabela de resumo salva em: {table_path}"
        )


if __name__ == "__main__":
    print(
        "Este script é projetado para ser usado "
        "como um módulo. Execute o arquivo main.py."
    )