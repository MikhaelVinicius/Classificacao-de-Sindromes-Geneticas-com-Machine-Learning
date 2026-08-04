import pandas as pd
import numpy as np

def flatten_data(raw_data):
    print("\n - Iniciando Pré-processamento de Dados - ")
    all_embeddings = []
    all_syndrome_ids = []
    all_subject_ids = []
    all_image_ids = []
    
    for syndrome_id, subjects in raw_data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                all_embeddings.append(embedding)
                all_syndrome_ids.append(syndrome_id)
                all_subject_ids.append(subject_id)
                all_image_ids.append(image_id)
                
    embeddings_array = np.vstack(all_embeddings)
    
    df = pd.DataFrame({
        'embedding': list(embeddings_array),
        'syndrome_id': all_syndrome_ids,
        'subject_id': all_subject_ids,
        'image_id': all_image_ids
    })
    
    print(f"Dados achatados em DataFrame com {len(df)} linhas e {len(df.columns)} colunas.")
    print("Colunas do DataFrame: ", df.columns.tolist())
    return df

def check_data_integrity(df):
    print("\n - Verificando Integridade dos Dados - ")
    print("Valores ausentes por coluna:")
    print(df.isnull().sum())
    
    embedding_dimensions = df['embedding'].apply(len).unique()
    if len(embedding_dimensions) > 1:
        print(f"Alerta: Embeddings com dimensões diferentes encontradas: {embedding_dimensions}")
    else:
        print(f"Todos os embeddings têm dimensão: {embedding_dimensions[0]}")
        
    if df['syndrome_id'].isnull().any() or (df['syndrome_id'] == '').any():
        print("Alerta: Valores nulos ou vazios encontrados em 'syndrome_id'.")
    else:
        print("Nenhum valor nulo ou vazio em 'syndrome_id'.")
        
    print("Integridade dos dados verificada.")

if __name__ == "__main__":
    from data_loader import load_data
    import os
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    processed_df = flatten_data(raw_data)
    check_data_integrity(processed_df)
    print("\nDataFrame resultante (primeiras 5 linhas):\n", processed_df.head())