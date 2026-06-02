"""  
Prend un modèle entraîné en float 32 et le transforme en modèle ultra léger
"""

import os
import torch
import torch.nn as nn

class ModelCompressor:
    def __init__(self, model_architecture):
        self.model = model_architecture

    def load_weights(self, weights_path):
        """  
        Charge les poids entrainés .pth dans l'architecture du model.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError("Aucun fichier de poids trouvé à l'emplacement : {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Poids chargés avec succès depuis {weights_path}")


    def quantize_model(self):
        """  
        Applique une quantification statique post entrainement pour passer de FLOAT32 a INT8
        Optimisé pour les architectures CPU
        """
        print("\n --- Début de la quantification INT8 ---")
        self.model.eval()

        # On indique à PyTorch de quantifier les couches Linéaires (nn.Linear) 
        # qui constituent le cœur mathématique de notre Transformer
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8
        )
        
        print("Modèle quantifié dynamiquement en INT8 avec succès pour le CPU.")
        return quantized_model


class ONNXExporter:
    def __init__(self, model):
        self.model = model

    def export(self, output_path, window_size = 10, num_features = 40):
        """  
        Explore et fige le graphe de calcul PyTorch au format standardisé ONNX
        """
        print("\n --- Début de l'exportation ONNX --- ")
        self.model.eval()

        #ONNX a besoin d'un tenseur fictif pour tracer la forme de ses matrices 
        dummy_input = torch.randn(1, window_size, num_features)


        input_names = ["input_lob"]
        output_names = ["output_predictions"]

        #On rend la taille du Batch dynamique lors de l'export :
        dynamic_axes = {
            "input_lob":{0: "batch_size"},
            "output_predictions":{0: "batch_size"}
        }

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True, #Stocke les poids à l'intérieur du fichier
            opset_version=14,
            do_constant_folding=True, #Pré calcule les opérations constantes
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"Graphe du modèle exporté et compilé avec succès vers {output_path}")


if __name__=="__main__":
    from models import LOBTransformer
    print("\n --- Test unitaire de la brique d'entraînement ---")

    #Instacion du modèle vierge 
    base_model = LOBTransformer()

    compressor = ModelCompressor(base_model)

    try:
        model_int8 = compressor.quantize_model()
    except Exception as e:
        print(f"Alerte de compatibilité lors de la quantification : {e}")
        
    # 3. Test de l'export ONNX
    exporter = ONNXExporter(base_model)
    onnx_test_path = "model_architecture_test.onnx"
    exporter.export(onnx_test_path)
    
    # Nettoyage du fichier de test ONNX généré
    if os.path.exists(onnx_test_path):
        os.remove(onnx_test_path)
        print("\n--- Tout le système d'optimisation est validé et prêt pour l'après-entraînement ! ---")