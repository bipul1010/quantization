import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from linear_quantization_per_channel_group import linear_quantization_per_channel


class W8A16Linear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, dtype=torch.float32
    ) -> None:
        super().__init__()

        self.register_buffer(
            "weight_int8",
            torch.randint(-128, 127, (out_features, in_features)).to(dtype=torch.int8),
        )

        self.register_buffer("scale", torch.rand(out_features, 1).to(dtype=dtype))

        if bias is True:
            self.register_buffer("bias", torch.rand(1, out_features).to(dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weight: torch.Tensor):
        wt_32 = weight.clone()

        quantized_wt, scale = linear_quantization_per_channel(r=wt_32, dim=0)
        self.weight_int8 = quantized_wt
        self.scale = scale
        # print(self.scale, self.scale.shape)

    def forward(self, x: torch.Tensor):
        # (b,seq,in_dim) * (in_dim,out_dim) -> (b,seq,out_dim)
        output = F.linear(input=x, weight=self.weight_int8.type_as(x))
        # dequantized_wt = self.weight_int8.float() * self.scale
        # output = x @ dequantized_wt.transpose(0, 1)
        # print(output)
        output = output * self.scale.transpose(0, 1)
        # print(output)
        if self.bias is not None:
            output = output + self.bias

        return output.type_as(x)


if __name__ == "__main__":
    import sys
    import time

    start_time = time.time()
    in_features, out_features = 4, 10
    torch.manual_seed(0)
    weight = torch.randn(out_features, in_features)
    print(f"Weight: {weight}")

    # x = torch.rand(2, 3, 4)
    x = torch.tensor(
        [
            [
                [0.1173, 0.0250, 0.7702, 0.2477],
                [0.9178, 0.3109, 0.0881, 0.6668],
                [0.9238, 0.7333, 0.4747, 0.3611],
            ],
            [
                [0.5182, 0.3115, 0.0384, 0.8643],
                [0.3944, 0.9342, 0.0652, 0.8689],
                [0.9224, 0.6649, 0.8659, 0.8462],
            ],
        ]
    )
    # print(x, bias)

    module = W8A16Linear(in_features, out_features, bias=True)
    # start_time = time.time()
    # module.quantize(weight=weight)
    start_time = time.time()
    # output = module(x)
    # print(output, output.dtype)

    output = F.linear(x, weight, bias=module.bias)
    print(output)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
