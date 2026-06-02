import pandas as pd
import numpy as np

def explore_data():
    path = "data_raw/train_us5.csv"
    print(f"Chargement de {path}...")
    
    # Lecture des premières lignes
    df = pd.read_csv(path)
    
    print("\n--- Informations Géométriques ---")
    print(f"Nombre de lignes (ticks) : {df.shape[0]}")
    print(f"Nombre de colonnes (features + labels) : {df.shape[1]}")
    
    # Affichage des premières colonnes pour comprendre la structure
    print("\n--- Les 5 premières colonnes du carnet ---")
    print(df.iloc[:5, :5])
    
    # Le dataset FI-2010 sous format CSV place généralement les labels à la fin.
    # Regardons les dernières colonnes pour localiser nos cibles boursières.
    print("\n--- Les 5 dernières colonnes (potentiellement les labels) ---")
    print(df.iloc[:5, -5:])
    
    # Distribution du label de prédiction principal (souvent la dernière colonne)
    last_col = df.columns[-1]
    print(f"\n--- Distribution des classes pour le label '{last_col}' ---")
    print(df[last_col].value_counts(normalize=True))

if __name__ == "__main__":
    explore_data()