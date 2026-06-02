import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LimitOrderBookDataset(Dataset):
    def __init__(self, csv_path, window_size=10, horizon_idx=-1):
        """
        Args:
            csv_path (str): Chemin vers le fichier CSV.
            window_size (int): Nombre de ticks passés à observer (historique).
            horizon_idx (int): Choix de l'horizon de prédiction (-1 pour le plus lointain, soit 1.0.4).
        """
        self.window_size = window_size
        
        # Chargement sans header car la première ligne contient déjà des données numériques
        df = pd.read_csv(csv_path, header=None)
        
        # Étape 1 : Séparer les features (40 premières colonnes) et les labels (5 dernières)
        X_raw = df.iloc[:, :40].values.astype(np.float32)
        y_raw = df.iloc[:, 40:].values
        
        # Étape 2 : Sélectionner l'horizon cible et adapter les labels pour PyTorch
        # Les fichiers ont des étiquettes [0.0, 1.0, 2.0].
        # PyTorch CrossEntropy exige des entiers stricts : 0 (Baisse), 1 (Stagnation), 2 (Hausse).
        target_labels = y_raw[:, horizon_idx].astype(np.int64)
        
        # Étape 3 : Vectorisation par fenêtres glissantes
        self.X = []
        self.y = []
        
        # On parcourt le dataset pour construire nos historiques
        for i in range(len(df) - window_size):
            # Fenêtre temporelle de taille (window_size, 40)
            window = X_raw[i : i + window_size]
            # Le label associé est celui situé au TOUT DERNIER tick de la fenêtre observée
            label = target_labels[i + window_size - 1]
            
            self.X.append(window)
            self.y.append(label)
            
        # Conversion finale en tenseurs PyTorch de type CPU natif
        self.X = torch.tensor(np.array(self.X))
        self.y = torch.tensor(np.array(self.y))
        
        print(f"Dataset {csv_path} chargé : {self.X.shape[0]} fenêtres glissantes créées.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Petit script de validation locale du bon fonctionnement
if __name__ == "__main__":
    # Test unitaire sur le fichier de validation
    dataset_test = LimitOrderBookDataset("data_raw/val.csv", window_size=10)
    
    # Création d'un DataLoader CPU standard
    data_loader = DataLoader(dataset_test, batch_size=32, shuffle=True)
    
    # Inspection de la géométrie du premier Batch d'inférence
    first_X, first_y = next(iter(data_loader))
    print(f"Forme du Batch de caractéristiques (Batch_size, Window_size, Features) : {first_X.shape}")
    print(f"Forme du Batch de cibles (Labels) : {first_y.shape}")