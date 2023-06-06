import torch.nn as nn

class Recommender(nn.Sequential):
    def __init__(self, n_movies: int, hidden_layer_neurons: int):
        """

        Args:
            n_movies (_type_): _description_
            n_users (_type_): _description_
            n_latent_factors (int, optional): _description_. Defaults to 20.
        """
        super().__init__()
        self.n_movies = n_movies
        self.add_module("inp", nn.Linear(self.n_movies, hidden_layer_neurons))
        self.add_module("inp_bn", nn.BatchNorm1d(hidden_layer_neurons))
        self.add_module("inp_relu", nn.LeakyReLU())

        self.add_module("hidden1", nn.Linear(hidden_layer_neurons, hidden_layer_neurons))
        self.add_module("bn1", nn.BatchNorm1d(hidden_layer_neurons))
        self.add_module("relu1", nn.LeakyReLU())

        self.add_module("hidden2", nn.Linear(hidden_layer_neurons, hidden_layer_neurons))
        self.add_module("bn2", nn.BatchNorm1d(hidden_layer_neurons))
        self.add_module("relu3", nn.LeakyReLU())

        self.add_module("hidden3", nn.Linear(hidden_layer_neurons, hidden_layer_neurons))
        self.add_module("bn3", nn.BatchNorm1d(hidden_layer_neurons))
        self.add_module("relu3", nn.LeakyReLU())

        self.add_module("hidden4", nn.Linear(hidden_layer_neurons, hidden_layer_neurons))
        self.add_module("bn4", nn.BatchNorm1d(hidden_layer_neurons))
        self.add_module("relu4", nn.LeakyReLU())

        self.add_module("outp", nn.Linear(hidden_layer_neurons, self.n_movies))
