from typing import List
import torch
import torch.nn as nn
from eight_bit_quantizer import W8A16Linear


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)
        self.w1 = nn.Linear(dim, 4 * dim)
        self.w2 = nn.Linear(4 * dim, dim)
        self.w3 = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        # x - > (b,seq) -> (b,seq,dim)
        embed_x = self.embed(x)

        # (b,seq,dim) -> (b,seq,vocab_size)
        return self.w3(self.w2(self.w1(embed_x)))

    def replace_with_linear_layer(
        self, target_module: nn.Module, module_name_to_exclude: List
    ):
        for name, module in self.named_children():
            if name in module_name_to_exclude:
                continue
            if isinstance(module, nn.Linear):
                module_bias = module.bias
                bias = True if module_bias is not None else False
                new_module = target_module(
                    module.in_features, module.out_features, bias, module.weight.dtype
                )
                new_module.bias = module_bias
                new_module.quantize(module.weight)
                setattr(self, name, new_module)


def orginal_output(module: nn.Module, input: torch.Tensor):
    return module(x=input)


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 10
    dim = 4
    x = torch.randint(0, vocab_size, (2, 3))

    module = DummyModel(vocab_size=vocab_size, dim=dim)

    print(module)

    for n, p in module.named_modules():
        print(n, p)
        if isinstance(p, nn.Linear):
            print(n, p.weight.dtype, p.weight.numel(), p.bias)
    # output = orginal_output(module=module, input=x)
    # print(output, output.shape, output.dtype, module.get_memory_footprint())
    print("========After replacement \n\n")
    module.replace_with_linear_layer(
        target_module=W8A16Linear, module_name_to_exclude=["w3"]
    )

    print(module)
    for n, p in module.named_children():
        try:
            print(n, p.weight.dtype, p.weight.numel())
        except:
            print(n, p.weight_int8.dtype, p.weight_int8.numel())

    # output = orginal_output(module=module, input=x)
    # print(output, output.shape, output.dtype)

    # for n, p in module.named_children():
    #     if n == "embed":
    #         print(n, type(n), p, isinstance(p, nn.Linear))
    #     else:
    #         print(n, type(n), p, isinstance(p, nn.Linear), p.bias)

    # output = orginal_output(module=module, input=x)
    # print(output, output.shape)
