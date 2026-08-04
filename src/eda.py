import pandas as pd
import numpy as np

def perform_eda(df):
    print("\n - Iniciando Análise Exploratória de Dados (EDA) - ")
    
    print("\nEstatísticas Gerais:")
    print(f"Número total de imagens (embeddings): {len(df)}")
    print(f"Número de síndromes únicas: {df['syndrome_id'].nunique()}")
    print(f"Número de sujeitos únicos: {df['subject_id'].nunique()}")
    
    print("\nDistribuição de Imagens por Síndrome:")
    syndrome_counts = df['syndrome_id'].value_counts().sort_index()
    print(syndrome_counts)
    
    min_images_per_syndrome = syndrome_counts.min()
    max_images_per_syndrome = syndrome_counts.max()
    print(f"\nNúmero mínimo de imagens por síndrome: {min_images_per_syndrome}")
    print(f"Número máximo de imagens por síndrome: {max_images_per_syndrome}")
    
    if max_images_per_syndrome / min_images_per_syndrome > 2:
        print("Alerta: O dataset apresenta um desequilíbrio significativo.")
        print("Isso pode impactar o desempenho do modelo e exigir técnicas de balanceamento.")
    else:
        print("O dataset parece razoavelmente balanceado em relação ao número de imagens.")
        
    subject_syndrome_check = df.groupby('subject_id')['syndrome_id'].nunique()
    subjects_with_multiple_syndromes = subject_syndrome_check[subject_syndrome_check > 1]
    
    if not subjects_with_multiple_syndromes.empty:
        print("\nAlerta: Sujeitos com múltiplas síndromes encontradas:")
        print(subjects_with_multiple_syndromes)
    else:
        print("Nenhum sujeito associado a múltiplas síndromes.")
        
    print("Análise Exploratória de Dados concluída.")

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessor import flatten_data, check_data_integrity
    import os
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    processed_df = flatten_data(raw_data)
    check_data_integrity(processed_df)
    perform_eda(processed_df)