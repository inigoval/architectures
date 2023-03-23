import torch.nn as nn

# Largely lifted from official repoe https://github.com/locuslab/convmixer/blob/47048118e95721a00385bfe3122519f4b583b26e/convmixer.py


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7):
    """
    Conv Mixer encoder


    :param dim: dimension of the input and output
    :param depth: number of layers
    :param kernel_size: kernel size of the convolutions
    :param patch_size: size of the patches
    """
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    # Depth-wise convolutions
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                # Combine depth-wise convolutions across channels with 1x1 convolutions
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
            for _ in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # nn.Linear(dim, n_classes)
    )
