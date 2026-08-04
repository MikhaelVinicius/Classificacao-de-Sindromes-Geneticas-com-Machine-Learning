import pickle
import numpy as np
import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo de dados não foi encontrado em: {file_path}")
    print(f"Carregando dados de: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Dados carregados com sucesso.")
    return data

def inspect_data_structure(data):
    print("\n - Estrutura dos Dados - ")
    if not data:
        print("O dicionário de dados está vazio.")
        return
    
    first_syndrome_id = next(iter(data.keys()))
    print(f"Primeiro Syndrome ID: {first_syndrome_id}")
    
    first_subject_id = next(iter(data[first_syndrome_id].keys()))
    print(f"Primeiro Subject ID para {first_syndrome_id}: {first_subject_id}")
    
    first_image_id = next(iter(data[first_syndrome_id][first_subject_id].keys()))
    embedding = data[first_syndrome_id][first_subject_id][first_image_id]
    print(f"Primeiro Image ID para {first_subject_id}: {first_image_id}")
    print(f"Dimensão do Embedding: {len(embedding)}")
    print(f"Tipo do Embedding: {type(embedding)}")
    print(f"Exemplo de Embedding (primeiros 5 valores): {embedding[:5]}")
    
    num_syndromes = len(data)
    num_subjects = sum(len(data[s]) for s in data)
    num_images = sum(sum(len(data[s][sub]) for sub in data[s]) for s in data)
    print(f"Número total de síndromes únicas: {num_syndromes}")
    print(f"Número total de sujeitos únicos: {num_subjects}")
    print(f"Número total de imagens (embeddings): {num_images}")

if __name__ == "__main__":
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    inspect_data_structure(raw_data)