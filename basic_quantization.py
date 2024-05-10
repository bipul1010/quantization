import torch


def clamp(params: torch.tensor, lower_bound: int, upper_bound: int):

    params = params.masked_fill(params < lower_bound, lower_bound)
    params = params.masked_fill(params > upper_bound, upper_bound)

    return params


def asymmetric_quantize(params: torch.tensor, bits: int):
    alpha = torch.max(params)
    beta = torch.min(params)

    scale = (alpha - beta) / (2**bits - 1)

    zero = torch.round(-1 * beta / scale)

    lower_bound, upper_bound = 0, 2**bits - 1

    quantized = clamp(
        params=torch.round(params / scale),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return quantized, scale, zero


def asymmetric_dequantize(
    quantized_params: torch.tensor, scale: torch.tensor, zero: torch.tensor
):

    dequantized = scale * (quantized_params - zero)
    return dequantized


def symmetric_quantize(params: torch.tensor, bits: int):
    alpha = torch.max(abs(params))

    scale = alpha / (2 ** (bits - 1) - 1)

    lower_bound, upper_bound = -(2 ** (bits - 1) - 1), (2 ** (bits - 1) - 1)
    print(lower_bound, upper_bound)

    quantized = clamp(
        params=torch.round(params / scale),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return quantized, scale


def symmetric_dequantize(params: torch.tensor, scale: torch.tensor):
    quantized = scale * params
    return quantized


if __name__ == "__main__":
    x = torch.rand(100) * 100
    x[0], x[1] = torch.max(x) + 1, torch.min(x) - 1

    print(x)

    # asymmetric
    # y = asymmetric_quantize(x, bits=8)
    # print(y[0])

    # m = asymmetric_dequantize(y[0], y[1], y[2])
    # print(m)

    ##symmetric

    y = symmetric_quantize(x, bits=8)
    print(y[0])

    m = symmetric_dequantize(y[0], y[1])
    print(m)
