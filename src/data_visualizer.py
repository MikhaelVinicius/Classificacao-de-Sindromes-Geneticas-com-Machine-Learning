import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_embeddings_tsne(df, output_dir='plots'):
    print("\n - Iniciando Visualização com t-SNE - ")
    
    X = np.array(df['embedding'].tolist())
    y = df['syndrome_id'].astype('category').cat.codes
    labels = df['syndrome_id'].values
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CORREÇÃO: 'max_iter' no lugar de 'n_iter'
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    print(f"Aplicando t-SNE com perplexity={tsne.perplexity}, max_iter={tsne.max_iter}")
    X_tsne = tsne.fit_transform(X)
    print("t-SNE concluído.")
    
    tsne_df = pd.DataFrame({
        'TSNE-1': X_tsne[:, 0],
        'TSNE-2': X_tsne[:, 1],
        'syndrome_id': labels
    })
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='TSNE-1', y='TSNE-2', hue='syndrome_id', data=tsne_df, palette='tab10',
        legend='full', alpha=0.7
    )
    plt.title('Visualização de Embeddings com t-SNE (Colorido por Syndrome ID)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(output_dir, 'tsne_embeddings_plot.png')
    plt.savefig(plot_path)
    print(f"Gráfico t-SNE salvo em: {plot_path}")
    plt.close()
    
    print("Visualização com t-SNE concluída.")

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessor import flatten_data
    import os
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    processed_df = flatten_data(raw_data)
    visualize_embeddings_tsne(processed_df)