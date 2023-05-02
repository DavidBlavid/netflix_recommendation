import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, n_movies, n_latent_factors=20):
        """

        Args:
            n_movies (_type_): _description_
            n_users (_type_): _description_
            n_latent_factors (int, optional): _description_. Defaults to 20.
        """
        super().__init__()
        self.n_movies = n_movies
        self.n_latent_factors = n_latent_factors
        self.encoder = nn.Sequential(
            nn.Linear(self.n_movies, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.n_latent_factors),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent_factors, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.n_movies),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    