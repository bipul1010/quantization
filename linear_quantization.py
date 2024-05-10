import torch


# r = s(q-z) where r is orginal tensor(float32) and q is quantized_tensor(int8);;; s is float32 and z is int8


def linear_quantization(
    r: torch.Tensor, scale: float, zero_point: int, dtype=torch.int8
):
    scaled_and_shifted_tensor = r / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

    quantized_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return quantized_tensor


def linear_dequantization(q: torch.Tensor, scale: float, zero_point: int):
    r = scale * (
        q.float() - zero_point
    )  # .float() is important because quantized_tensor is int8 and few parameters after subtracting from zero_point overflows between -128 to 127.
    return r


def get_scale_and_zero_point_asymmetric(r: torch.Tensor, dtype=torch.int8):

    """this is asymmetric mode"""

    ##r_min = s(q_min - z) ; r_max = s(q_max - z)

    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

    r_min, r_max = torch.min(r).item(), torch.max(r).item()

    ##scale
    s = (r_max - r_min) / (q_max - q_min)  ##from the first line comment

    # zero_point
    z = q_min - (r_min / s)  ## from r = s(q-z)

    if z < q_min:
        z = q_min
    elif z > q_max:
        z = q_max
    else:
        z = int(round(z))

    return (s, z)


# def linear_quantization_with_asymmetric_mode(r: torch.Tensor, dtype=torch.int8):
#     scale, zero_point = get_scale_and_zero_point(r=r)
#     scaled_and_shifted_tensor = r / scale + zero_point
#     rounded_tensor = torch.round(scaled_and_shifted_tensor)

#     q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

#     quantized_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
#     return quantized_tensor, scale, zero_point


# def linear_dequantization_with_asymmetric_mode(
#     q: torch.Tensor, scale: float, zero_point: int
# ):
#     r = scale * (q.float() - zero_point)
#     return r


def get_scale_and_zero_point_symmetric(r: torch.Tensor, dtype=torch.int8):
    """this is symmetric mode where zero_point=0"""
    q_max = torch.iinfo(dtype).max
    r_max = r.abs().max().item()

    s = r_max / q_max
    return s


# def linear_quantization_with_symmetric_mode(r: torch.Tensor, dtype=torch.int8):
#     s = get_scale(r=r)
#     q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

#     q = torch.round(r / s)
#     q = q.clamp(q_min, q_max).to(dtype=dtype)

#     return q, s


# def linear_dequantization_with_symmetric_mode(q: torch.Tensor, scale: float):
#     return scale * q.float()


if __name__ == "__main__":
    ### a dummy tensor to test the implementation
    test_tensor = torch.tensor(
        [[191.6, -13.5, 728.6], [92.14, 295.5, -184], [0, 684.6, 245.5]]
    )

    # test_tensor = torch.tensor(
    #     [
    #         [-0.0029, 0.4210, -1.0896, -0.9137],
    #         [1.1656, 0.4902, 2.6828, -1.9845],
    #         [-1.0135, -1.4527, 0.8607, -1.8480],
    #         [1.0863, -0.3104, -0.8486, -0.2968],
    #     ]
    # )
    # test_tensor = torch.rand(4, 5)
    print(test_tensor)

    ##asymmetric_mode
    scale, zero_point = get_scale_and_zero_point_asymmetric(r=test_tensor)
    quantized_tensor = linear_quantization(
        r=test_tensor, scale=scale, zero_point=zero_point
    )
    dequantized_tensor = linear_dequantization(
        q=quantized_tensor, scale=scale, zero_point=zero_point
    )
    print(
        f"Asymmetric Mode:: | Quantized_Tensor: {quantized_tensor} | Dequantized_Tensor:{dequantized_tensor}"
    )
    asymmetric_loss = (test_tensor - dequantized_tensor).square().mean()

    ## symmetric mode
    scale = get_scale_and_zero_point_symmetric(r=test_tensor)  ##zero_point will be 0
    quantized_tensor = linear_quantization(r=test_tensor, scale=scale, zero_point=0)
    dequantized_tensor = linear_dequantization(
        q=quantized_tensor, scale=scale, zero_point=0
    )
    print(
        f"Symmetric Mode:: | Quantized_Tensor: {quantized_tensor} | Dequantized_Tensor:{dequantized_tensor}"
    )
    symmetric_loss = (test_tensor - dequantized_tensor).square().mean()

    print(f"Loss:: Asymmetricloss:{asymmetric_loss} | Symmetric loss: {symmetric_loss}")
