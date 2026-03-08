import torch
import torch.nn as nn


# -----------------------------
# Double Convolution Block
# -----------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# U-Net Colorization Model
# -----------------------------
class UNetColorization(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Decoder
        self.up1 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)

        # Output layer (predict A,B channels)
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):

        # Encoder
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)

        # Decoder
        u1 = self.up(c3)
        u1 = torch.cat([u1, c2], dim=1)
        c4 = self.up1(u1)

        u2 = self.up(c4)
        u2 = torch.cat([u2, c1], dim=1)
        c5 = self.up2(u2)

        # Output (A,B channels)
        out = self.final(c5)

        return torch.tanh(out)