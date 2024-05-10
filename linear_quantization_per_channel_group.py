import torch


def linear_quantization_per_channel(r: torch.Tensor, dtype=torch.int8, dim=-1):
    """symmetric:per channel quantization is basically quantization across by rows if dim=0 , else columns if dim=1"""

    if dim < 0:
        raise ValueError("dim must be in (0,1)")
    n_rows, n_colns = r.shape

    output_dim = n_rows if dim == 0 else n_colns
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

    scale = torch.zeros(output_dim)  # (n,)

    for i in range(output_dim):
        r_max = r[i, :].abs().max().item() if dim == 0 else r[:, i].abs().max().item()
        scale[i] = r_max / q_max

    # reshaping scale
    scale = scale.unsqueeze(1) if dim == 0 else scale.unsqueeze(0)
    quantized_vector = r / scale

    quantized_vector = torch.round(quantized_vector).clamp(q_min, q_max).to(dtype=dtype)

    return quantized_vector, scale


def linear_dequantization_per_channel(q: torch.Tensor, scale: torch.Tensor):
    return scale * q.float()


def linear_quantization_per_group(r: torch.Tensor, group_size=-1):

    _, n_columns = r.shape

    assert n_columns % group_size == 0

    r_shape = r.view(-1, group_size)  # (n_rows * n_columns / group_size,group_size)
    quantized_vector, scale = linear_quantization_per_channel(
        r=r_shape, dim=0
    )  # (n_rows * n_columns / group_size,group_size) and scale: (n_rows * n_columns / group_size,1)

    quantized_vector = quantized_vector.view(*r.shape)

    return quantized_vector, scale


def linear_dequantization_per_group(
    q: torch.Tensor, scale: torch.Tensor, group_size=-1
):

    q_shape = q.view(-1, group_size)  # (n_rows * n_columns / group_size,group_size)
    dq = scale * q_shape.float()  # (n_rows * n_columns / group_size,group_size)

    return dq.view(*q.shape)


if __name__ == "__main__":
    test_tensor = torch.tensor(
        [
            [191.6, -13.5, 728.6, 216.5],
            [92.14, 295.5, -184, 220.5],
            [0, 684.6, 245.5, 316.5],
        ]
    )
    print(test_tensor)
    group_size = 2
    q, s = linear_quantization_per_channel(r=test_tensor, dim=0)

    # q, s = linear_quantization_per_group(r=test_tensor, group_size=group_size)

    dq = linear_dequantization_per_channel(q=q, scale=s)
    # dq = linear_dequantization_per_group(q=q, scale=s, group_size=group_size)
    loss = (test_tensor - dq).square().mean()
    print(
        f"Quantized Tensor: {q} |\n Scale: {s}|\n Dequantized Tensor: {dq}|\n Loss: {loss}"
    )
