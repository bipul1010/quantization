import torch.nn as nn


class BitNet(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool, device, dtype
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        self.nn_linear = nn.Linear(in_features, out_features)

        weight = self.nn_linear.weight
        gamma = weight.abs().mean()
