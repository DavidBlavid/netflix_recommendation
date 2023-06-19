import torch
from torch import nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.Dropout(0.8)

        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.SELU(),

            nn.Dropout(0.8)

        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),

            nn.Linear(256, input_size),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)