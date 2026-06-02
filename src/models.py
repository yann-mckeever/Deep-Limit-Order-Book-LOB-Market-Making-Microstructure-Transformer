import torch
import torch.nn as nn 

class LOBTransformer(nn.Module):
    def __init__(self, window_size=10, num_features=40, num_classes=3, use_attention=True):
        super().__init__()
        #BLOC 1
        #Extraction des caractéristiques spatiales du carnet
        #On utilise des convolutions 2D en adpatant la forme des tenseurs
        self.cnn_layer = nn.Sequential(
            #Conv2d attend : (Batch, Channels, Height, Width)
            #On va traiter notre fenêtre comme une image de taille (1, Window_size, Features)
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,2),stride=(1,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,2),stride=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        #Après le CNN, notre dimension 'Features' de 40 a été réduite à 10, multiplée par 32 channels
        #On projette cela vers une dimension fixe (d_model) pour le Transformer
        self.spatial_projection=nn.Linear(32*10,64)

        #BLOC Transformer
        #Capture la dynamique temporelle (microstructure)
        #On définit un encodeur léger à 2 couches et 4 têtes d'attention pour le CPU

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(64,num_classes)


    def forward(self, x):
        # x shape (Batch, Window size, Features)

        #On ajoute une dimension "Channel=1" pour simuler une image pour le CNN
        x=x.unsqueeze(1) #Shape devient (Batch, 1, Window size, Features)

        #Passage dans le CNN
        x=self.cnn_layer(x) #Shape : (Batch, 32, Window size, 10)

        #Permutation et applatissement pour préparer l'entrée dans le transformer
        batch_size, channels, window_size, feat_out = x.shape
        x = x.permute(0,2,1,3).contiguous() #Shape : (Batch,  Window size, 32, 10)
        x = x.view(batch_size, window_size, channels*feat_out) #Shape : (Batch,  Window size, 320)

        #Projection vers d_model (64)
        x = self.spatial_projection(x) #Shape : (Batch,  Window size, 64)

        #Passage dans le Transformer
        x = self.transformer_encoder(x)  #Shape : (Batch,  Window size, 64)

        #On extrait uniquement le dernier état temporel de la fenêtre (tick le plus récent)
        last_time_step = x[:,-1,:] #Shape : (Batch, 64)

        output=self.classifier(last_time_step) #Shape : (Batch, 3)
        return output


if __name__=="__main__":
    #Teste à vide pour vérifier qu'aucune dimension ne plante à vide sur le CPU
    model=LOBTransformer()
    dummy_input=torch.randn(32,10,40)
    out=model(dummy_input)
    print(f"Test réussi ! Forme du tenseur de sortie : {out.shape}") # Doit afficher [32, 3]