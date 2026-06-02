import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Importation de nos propres briques POO
from src.dataset import LimitOrderBookDataset
from src.models import LOBTransformer
from src.pipeline import LOBTrainer, LOBEvaluator

def main():
    print("==================================================")
    
    # 1. HYPERPARAMÈTRES DE CONFIGURATION
    WINDOW_SIZE = 10
    BATCH_SIZE = 64       
    LEARNING_RATE = 0.001
    EPOCHS = 3             # 3 épochs pour tester la convergence sans surchauffer ton PC
    MODEL_SAVE_PATH = "best_lob_transformer.pth"
    
    # 2. CHARGEMENT ET DÉCOUPAGE DES DONNÉES (DATA LAYER)
    print("Étape 1 : Préparation et fenêtrage des datasets...")
    train_dataset = LimitOrderBookDataset("data_raw/train_us5.csv", window_size=WINDOW_SIZE)
    val_dataset = LimitOrderBookDataset("data_raw/val.csv", window_size=WINDOW_SIZE)
    test_dataset = LimitOrderBookDataset("data_raw/test.csv", window_size=WINDOW_SIZE)
    
    # Création des DataLoaders pour la distribution par lots (Batches)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. INITIALISATION DE L'ARCHITECTURE (MODEL LAYER)
    print("\nÉtape 2 : Initialisation du réseau CNN-Transformer...")
    model = LOBTransformer(window_size=WINDOW_SIZE, num_features=40, num_classes=3)
    
    # 4. CONFIGURATION DES OUTILS MATHÉMATIQUES
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. ENTRAÎNEMENT (PIPELINE LAYER)
    print("\nÉtape 3 : Démarrage de l'entraînement sur CPU...")
    trainer = LOBTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion
    )
    
    # Lancement de la boucle fit
    history = trainer.fit(epochs=EPOCHS)
    
    # Sauvegarde des poids entraînés (indispensable pour optimization.py plus tard !)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[Succès] Poids du modèle sauvegardés sous : {MODEL_SAVE_PATH}")
    
    # 6. ÉVALUATION FINALE (DIAGNOSTIC QUANT)
    print("\nÉtape 4 : Évaluation des performances sur le Test Set (Données futures)...")
    evaluator = LOBEvaluator(model=model, test_loader=test_loader)
    evaluator.evaluate()
    

if __name__ == "__main__":
    main()