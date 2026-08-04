import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import euclidean_distances, pairwise_distances

class CustomKNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
        if self.metric == 'euclidean':
            self.distance_func = euclidean_distances
        elif self.metric == 'cosine':
            self.distance_func = lambda X1, X2: 1 - pairwise_distances(X1, X2, metric='cosine')
        else:
            raise ValueError("Métrica suportada: 'euclidean' ou 'cosine'")
            
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict_proba(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("O classificador não foi treinado. Chame fit()")
            
        num_test_samples = X_test.shape[0]
        num_classes = len(np.unique(self.y_train))
        probabilities = np.zeros((num_test_samples, num_classes))
        
        for i, test_sample in enumerate(X_test):
            if self.metric == 'euclidean':
                distances = self.distance_func(test_sample.reshape(1, -1), self.X_train)[0]
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
            elif self.metric == 'cosine':
                similarities = self.distance_func(test_sample.reshape(1, -1), self.X_train)[0]
                nearest_indices = np.argsort(similarities)[-self.n_neighbors:]
                
            nearest_labels = self.y_train[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            
            for label, count in zip(unique_labels, counts):
                probabilities[i, label] = count / self.n_neighbors
                
        return probabilities
        
    def predict(self, X_test):
        probabilities = self.predict_proba(X_test)
        return np.argmax(probabilities, axis=1)

def prepare_data_for_classification(df):
    print("\n - Preparando Dados para Classificação - ")
    X = np.array(df["embedding"].tolist())
    
    unique_syndromes = df["syndrome_id"].astype("category").cat.categories
    y = df["syndrome_id"].astype("category").cat.codes.values
    
    print(f"Formato de X (embeddings): {X.shape}")
    print(f"Formato de y (labels): {y.shape}")
    print(f"Número de classes únicas: {len(unique_syndromes)}")
    print("Mapeamento de classes: ", dict(enumerate(unique_syndromes)))
    return X, y, unique_syndromes

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessor import flatten_data
    import os
    
    DATA_FILE_PATH = os.path.join('data', 'mini_gm_public_v0.1.p')
    raw_data = load_data(DATA_FILE_PATH)
    processed_df = flatten_data(raw_data)
    X, y, unique_syndromes = prepare_data_for_classification(processed_df)
    
    print("\n - Testando CustomKNeighborsClassifier (Euclidiana) -")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_euclidean = CustomKNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_euclidean.fit(X_train, y_train)
    predictions_euclidean = knn_euclidean.predict(X_test)
    probabilities_euclidean = knn_euclidean.predict_proba(X_test)
    print("Primeiras 5 previsões (Euclidiana):", predictions_euclidean[:5])
    print("Primeiras 5 probabilidades (Euclidiana):\n", probabilities_euclidean[:5])
    
    print("\n - Testando CustomKNeighborsClassifier (Cosseno) -")
    knn_cosine = CustomKNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_cosine.fit(X_train, y_train)
    predictions_cosine = knn_cosine.predict(X_test)
    probabilities_cosine = knn_cosine.predict_proba(X_test)
    print("Primeiras 5 previsões (Cosseno):", predictions_cosine[:5])
    print("Primeiras 5 probabilidades (Cosseno):\n", probabilities_cosine[:5])
    
    print("\n - Testando KNeighborsClassifier do scikit-learn (Euclidiana) -")
    knn_sklearn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_sklearn_euclidean.fit(X_train, y_train)
    predictions_sklearn_euclidean = knn_sklearn_euclidean.predict(X_test)
    probabilities_sklearn_euclidean = knn_sklearn_euclidean.predict_proba(X_test)
    print("Primeiras 5 previsões (scikit-learn Euclidiana):", predictions_sklearn_euclidean[:5])
    print("Primeiras 5 probabilidades (scikit-learn Euclidiana):\n", probabilities_sklearn_euclidean[:5])
    
    print("\n - Testando KNeighborsClassifier do scikit-learn (Cosseno) -")
    knn_sklearn_cosine = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_sklearn_cosine.fit(X_train, y_train)
    predictions_sklearn_cosine = knn_sklearn_cosine.predict(X_test)
    probabilities_sklearn_cosine = knn_sklearn_cosine.predict_proba(X_test)
    print("Primeiras 5 previsões (scikit-learn Cosseno):", predictions_sklearn_cosine[:5])
    print("Primeiras 5 probabilidades (scikit-learn Cosseno):\n", probabilities_sklearn_cosine[:5])