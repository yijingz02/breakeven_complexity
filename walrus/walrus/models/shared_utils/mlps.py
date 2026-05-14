import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SubsampledLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag.
    It takes as input:
    - input values
    - list of indices denoting which state variables from the state vocab are present
    It only performs the linear layer on rows/cols relevant to those state variables.
    It typically corresponds to an embedding bag for multiple datasets.
    Each dataset will have its own embedding.
    Embedding bags are gather in a unique linear module for reusability.
    The same model can thus be easily tested on different datasets,
    without having to change the embedding layers.


    Assumes (... C) input
    """

    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, labels):
        # Note - really only works if all batches are the same input type
        labels = labels
        label_size = len(labels)
        if self.subsample_in:
            scale = (
                (self.dim_in / label_size) ** 0.5
            )  # Equivalent to swapping init to correct for given subsample of input
            x = scale * F.linear(x, self.linear.weight[:, labels], self.linear.bias)
        else:
            x = F.linear(x, self.linear.weight[labels], self.linear.bias[labels])
        return x
