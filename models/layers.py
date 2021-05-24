import torch.nn as nn
def _prediction_mlp(in_dims: int, 
                    h_dims: int, 
                    out_dims: int) -> nn.Sequential:
    """Prediction MLP. The original paper's implementation has 2 layers, with 
    BN applied to its hidden fc layers but no BN or ReLU on the output fc layer.
    Note that the hidden dimensions should be smaller than the input/output 
    dimensions (bottleneck structure). The default implementation using a 
    ResNet50 backbone has an input dimension of 2048, hidden dimension of 512, 
    and output dimension of 2048
    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers (should be a
            bottleneck!)
        out_dims: 
            Output Dimension of the final linear layer.
    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    num_layers: int = 3) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with 
    BN applied to its hidden fc layers but no ReLU on the output fc layer. 
    The CIFAR-10 study used a MLP with only two layers.
    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers.
        out_dims: 
            Output Dimension of the final linear layer.
        num_layers:
            Controls the number of layers; must be 2 or 3. Defaults to 3.
    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection