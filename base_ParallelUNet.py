import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, channels):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return self.gamma * x + self.beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.sqrt_d = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.sqrt_d, dim=-1)
        return attn_weights @ V

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.sqrt_d = embed_dim ** 0.5

    def forward(self, z, Ic):
        Q = self.query(z)
        K = self.key(Ic)
        V = self.value(Ic)
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.sqrt_d, dim=-1)
        return attn_weights @ V

class ParallelUNet(nn.Module):
    def __init__(self):
        super(ParallelUNet, self).__init__()

        # Encoding path
        self.enc1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.enc2 = nn.ModuleList([ResidualBlock(128, 128) for _ in range(3)])
        self.enc3 = nn.ModuleList([ResidualBlock(128, 64) for _ in range(4)])
        self.cross_attn = CrossAttention(64)

        # Decoding path (for simplicity, just a few layers)
        self.dec1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

        # For pose embedding
        self.fc_person_pose = nn.Linear(128, 128)  # Example size
        self.fc_garment_pose = nn.Linear(128, 128)  # Example size

        # FiLM layers
        self.film1 = FiLM(128)

        # Self attention layer
        self.self_attn = SelfAttention(128)


    def forward(self, clothing_agnostic, noisy_image, person_pose, garment_pose):
        # Initial convolutions
        x1 = self.enc1(clothing_agnostic)
        x2 = self.enc1(noisy_image)
        x3 = self.enc1(garment_pose)

        # Person pose embedding
        person_pose_emb = self.fc_person_pose(person_pose)

        # Garment pose embedding
        garment_pose_emb = self.fc_garment_pose(garment_pose)

        # Encoding path
        for layer in self.enc2:
            x1 = layer(x1)
            x2 = layer(x2)
            x3 = layer(x3)

        # Apply self attention on x2 after initial convolutions
        x2 = self.self_attn(x2)

        # Cross attention
        x_attn = self.cross_attn(x2, x3)
        x2 += x_attn  # Add attention results to x2

        # Decoding
        x2 = self.dec1(x2)

        # Applying FiLM
        x2 = self.film1(x2)

        # Final convolution to produce the output
        out = nn.Conv2d(128, 3, kernel_size=3, padding=1)(x2)

        return out

model = ParallelUNet()
