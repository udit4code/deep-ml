from tinygrad import Tensor

class MyTransform:
    def __call__(self, x: Tensor) -> Tensor:
        """
        x: tinygrad Tensor of shape (1, 28, 28), float, in [0, 1].
        Return: transformed Tensor, same shape and dtype.
        Must be non-identity and deterministic under Tensor.manual_seed.
        """
        # TODO: implement your custom transformation logic here
        # Brighten slightly and clamp back to [0, 1]
        return (x * 0.9 + 0.05).clip(0.0, 1.0)
