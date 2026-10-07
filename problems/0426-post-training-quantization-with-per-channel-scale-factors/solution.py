import torch

def per_channel_quantize(weight: torch.Tensor, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix tensor of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int tensor of shape (out_channels, in_features)
        - scale_factors: float tensor of shape (out_channels,)
        - dequantized_weights: float tensor of shape (out_channels, in_features)
    """
    if weight.ndim != 2:
        raise ValueError(
            f"Expected a 2D weight matrix, got shape {weight.shape}"
        )

    if bits < 2:
        raise ValueError("bits must be >= 2")

    # Example for INT8:
    # qmax = 2^7 - 1 = 127
    # Quantization range = [-127, 127]
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax

    # --------------------------------------------------------
    # 1. Find maximum absolute value independently per channel.
    #
    # weight.shape = [out_channels, in_features]
    # max_abs.shape = [out_channels]
    # --------------------------------------------------------
    max_abs = weight.abs().amax(dim=1)

    # --------------------------------------------------------
    # 2. Compute one scale per output channel.
    #
    # scale[c] = max(|W[c, :]|) / qmax
    #
    # Handle an all-zero channel specially to avoid scale = 0.
    # --------------------------------------------------------
    scale = max_abs / qmax
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale,
    )

    # --------------------------------------------------------
    # 3. Quantize.
    #
    # scale[:, None] changes:
    #     [out_channels]
    # into
    #     [out_channels, 1]
    #
    # so it broadcasts across in_features.
    # --------------------------------------------------------
    quantized = torch.round(
        weight / scale[:, None]
    ).clamp(qmin, qmax)

    # Use an integer storage type.
    if bits <= 8:
        quantized = quantized.to(torch.int8)
    elif bits <= 16:
        quantized = quantized.to(torch.int16)
    else:
        quantized = quantized.to(torch.int32)

    # --------------------------------------------------------
    # 4. Dequantize.
    #
    # W_hat[c, j] = Q[c, j] * scale[c]
    # --------------------------------------------------------
    dequantized = quantized.to(weight.dtype) * scale[:, None]

    return quantized, scale, dequantized