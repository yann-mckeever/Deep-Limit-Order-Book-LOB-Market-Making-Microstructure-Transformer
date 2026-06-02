import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix



class LOBTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        """  
        Classe d'orchestration de l'entraînement sur CPU
        """
        self.model=model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self,epoch_idx):
        """  
        Effectue une phase d'entraînement complète sur le CPU
        """
        self.model.train() #Active le dropout et le BatchNorm
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar=tqdm(self.train_loader,desc=f"epoch {epoch_idx +1} [Train]")

        for X_batch, y_batch in progress_bar :
            # RAZ des gradients calculés par le batch précédent
            self.optimizer.zero_grad()

            #Forward pass prédiction du modèle
            outputs = self.model(X_batch)

            #Cacul de la perte
            loss = self.criterion(outputs, y_batch)

            #Backward pass : calcul auto des gradients
            loss.backward()

            #MAJ des poids par l'optimizer
            self.optimizer.step()

            #Métriques de suivi intermédiaires
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            #MAJ de la barre de progréssion
            progress_bar.set_postfix(loss=loss.item(), acc = 100.0 * correct / total)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch_idx):
        """  
        Evalue le compromis Biais - Variance sur le dataset de validation
        """
        self.model.eval() #Désactive le droupout et fige le BatchNorm
        #Calcul la loss et l'accuracy de validation sans calcul de gradient
        running_loss = 0.0
        correct = 0 
        total = 0

        #torch.no_grad() coupe le calcul du gradient pour économiser la mémoire du CPU
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch_idx+1} [Val]  -> Loss : {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        return val_loss, val_acc
    


    def fit(self, epochs):
        """  
        Boucle principale qui orchestre le run complet
        """

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print('-'*50)
        return history

    
class LOBEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

        
    def evaluate(self):
        """  
        Génère un rapport de classification complet (Precision, Recall, F1) sur le test set.
        """

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs=self.model(X_batch)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.numpy())
                all_labels.extend(y_batch.numpy())

        print("\n" + "="*20 + "RAPPORT D'EVALUATION FINALE" + "="*20)
        #Rapport sklearn pour analyser finement chaque classe (Baisse, Stagantion, Hausse)
        print(classification_report(all_labels, all_preds, target_names=['Baisse (0)', 'Stagnation (1)', 'Hausse (2)']))
        print("Matrice de confusion :")
        print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    from models import LOBTransformer  # On importe ton modèle validé
    
    print("--- Démarrage du test unitaire de pipeline.py ---")
    
    # 1. Configuration des dimensions fictives (Simule 64 exemples de marché)
    batch_size = 16
    window_size = 10
    num_features = 40
    num_classes = 3
    
    # Création de faux tenseurs PyTorch de la bonne forme
    X_dummy = torch.randn(64, window_size, num_features)
    # 64 étiquettes aléatoires entre 0 (Baisse), 1 (Stagnation) et 2 (Hausse)
    y_dummy = torch.randint(0, num_classes, (64,))
    
    # 2. Création de faux DataLoaders (Train, Val, Test)
    dummy_dataset = TensorDataset(X_dummy, y_dummy)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialisation des briques PyTorch
    model = LOBTransformer(window_size=window_size, num_features=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Test du LOBTrainer (sur 2 époques rapides)
    print("\n[Test] Initialisation du Trainer...")
    trainer = LOBTrainer(
        model=model, 
        train_loader=dummy_loader, 
        val_loader=dummy_loader, 
        optimizer=optimizer, 
        criterion=criterion
    )
    
    print("[Test] Lancement de fit() pour 2 époques...")
    history = trainer.fit(epochs=2)
    print("Historique d'entraînement généré :", history)
    
    # 5. Test du LOBEvaluator
    print("\n[Test] Initialisation de l'Evaluator...")
    evaluator = LOBEvaluator(model=model, test_loader=dummy_loader)
    evaluator.evaluate()
    
    print("\n--- Tout le pipeline fonctionne à merveille ! ---")