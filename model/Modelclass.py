import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DroneAcoustics(nn.Module):
    def __init__(self):
        super().__init__()  # hérite de la classe nn.Module
        self.flatten = nn.Flatten() # "applatis" les images en un vecteur
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # couche linéaire (taille entrée, taille sortie)
            nn.ReLU(), # fonction d'activation
            nn.Linear(512, 512), # couche linéaire
            nn.ReLU(), # fonction d'activation
            nn.Linear(512, 2) # couche linéaire 
        )

    def forward(self, x): # fonction forward qui définit comment les données passent dans le réseau
        x = self.flatten(x) # applatit les images
        logits = self.linear_relu_stack(x) # passe les données dans le réseau
        return logits

