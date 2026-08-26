# Genetic Syndrome Classification with Machine Learning

## Project Description

The main objective is to analyze image embeddings (320-dimensional feature vectors) to classify genetic syndromes.

The complete pipeline covers the entire workflow, from processing nested data stored in a `.pickle` file to performing Exploratory Data Analysis (EDA), visualizing the embeddings using t-SNE, and implementing a custom K-Nearest Neighbors (KNN) classifier. The model was evaluated using 10-fold cross-validation, comparing Euclidean and Cosine distance metrics while computing key performance metrics such as Accuracy, F1-Score, AUC, and Top-3 Accuracy.

---

## Project Structure

The project is organized as follows:

```text
├── data/
│   └── mini_gm_public_v0.1.p        # Original dataset (Pickle)
├── plots/                           # Automatically generated output directory
│   ├── k_optimization_*.png         # K optimization plots (Accuracy, F1, AUC, Top-3)
│   ├── performance_summary_table.md # Final performance summary table
│   ├── roc_curve_*.png              # ROC curves (Euclidean, Cosine and comparison)
│   └── tsne_embeddings_plot.png     # t-SNE visualization of embeddings
├── src/                             # Python source code
│   ├── data_loader.py               # Data loading and inspection
│   ├── data_preprocessor.py         # Data flattening and integrity checks
│   ├── data_visualizer.py           # Dimensionality reduction (t-SNE)
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── knn_classifier.py            # Custom KNN implementation
│   ├── main.py                      # Main pipeline execution
│   ├── metrics_calculator.py        # Manual implementation of evaluation metrics
│   ├── model_evaluator.py           # 10-fold cross-validation and hyperparameter search
│   └── results_visualizer.py        # Generation of plots and summary tables
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## Requirements

- Python 3.8 or higher

It is recommended to use a virtual environment before installing the dependencies.

Install all required packages by running:

```bash
pip install -r requirements.txt
```

> **Note:** The `tabulate` package is included to generate properly formatted Markdown tables in the terminal.

---

## Running the Project

The project was designed to execute the entire Machine Learning pipeline through a single entry point, without relying on Jupyter Notebooks.

From the project's root directory, run:

```bash
python src/main.py
```

---

## Execution Pipeline

Running the command above will automatically execute the following steps:

1. **Data Loading**
   - Loads the `.pickle` dataset from the `data/` directory.

2. **Preprocessing & Exploratory Data Analysis**
   - Flattens the nested data structure.
   - Performs data integrity validation.
   - Computes descriptive statistics and class distribution.

3. **Visualization**
   - Generates a two-dimensional t-SNE projection of the embeddings.

4. **Model Training & Evaluation**
   - Trains a custom K-Nearest Neighbors classifier.
   - Performs 10-fold stratified cross-validation.
   - Searches for the optimal value of **k** (from 1 to 15), using **F1-Score** as the optimization criterion.
   - Evaluates both **Euclidean** and **Cosine** distance metrics.

5. **Results Generation**
   - Generates ROC curves.
   - Produces K optimization plots.
   - Creates the final performance summary table.
   - Saves all generated outputs in the `plots/` directory.
