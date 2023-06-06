import torch.nn as nn

class AutoEncoder(nn.Sequential):
    def __init__(self, n_movies: int):
        """

        Args:
            n_movies (_type_): _description_
        """
        super().__init__()
        self.n_movies = n_movies

        self.encoder_net = nn.Sequential(
            nn.Linear(self.n_movies, 4096),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),

            nn.Linear(4096, self.n_movies),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder_net(x)
    
    def decode(self, x):
        return self.decoder_net(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))