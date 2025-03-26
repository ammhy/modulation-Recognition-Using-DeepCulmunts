import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class EnhancedWaveletDenoise(nn.Module):
    def __init__(self, wavelet='db4', levels=2):
        super().__init__()
        self.dwt = DWT1DForward(wave=wavelet, J=levels)
        self.idwt = DWT1DInverse(wave=wavelet)
        self.levels = levels
        # 分层可学习阈值
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(levels)
        ])
    def adaptive_threshold(self, coeff, level):
        """自适应软阈值函数"""
        threshold = self.thresholds[level]
        return torch.sign(coeff) * torch.relu(torch.abs(coeff) - threshold)
    def forward(self, x):
        # 输入形状: [B, T, 2]
        x = x.permute(0, 2, 1)  # [B, 2, T]
        # 小波分解与重构
        yl, yh = self.dwt(x)
        for l in range(self.levels):
            yh[l] = self.adaptive_threshold(yh[l], l)
        denoised = self.idwt((yl, yh))[:, :, :x.shape[-1]]
        return denoised.permute(0, 2, 1)  # 恢复原始形状[B, T, 2]


class ResNetTransformer(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # 信号预处理
        self.wavelet_denoise = EnhancedWaveletDenoise()

        # ResNet主干
        self.resnet = nn.Sequential(
            nn.Conv1d(2, 64, 7, 2, 3),  # [B,64,T/2]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self._make_res_layer(64, 64, 2),  # [B,64,T/2]
            self._make_res_layer(64, 128, 2, stride=2),  # [B,128,T/4]
            self._make_res_layer(128, 256, 2, stride=2),  # [B,256,T/8]
            self._make_res_layer(256, 512, 2, stride=2),  # [B,512,T/16]
        )

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=3
        )

        # 特征融合
        self.fc = nn.Sequential(
            nn.Linear(512 + 2, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def _make_res_layer(self, in_c, out_c, blocks, stride=1):
        """构建残差层"""
        layers = []
        layers.append(ResidualBlock1D(in_c, out_c, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_c, out_c))
        return nn.Sequential(*layers)

    def _calc_cumulants(self, x):
        """计算关键累积量"""
        I = x[:, :, 0] - torch.mean(x[:, :, 0], dim=1, keepdim=True)
        Q = x[:, :, 1] - torch.mean(x[:, :, 1], dim=1, keepdim=True)
        return torch.stack([
            torch.mean(I * Q ** 2, dim=1),  # C20
            torch.mean(I ** 2 * Q, dim=1)  # C21
        ], dim=1)

    def forward(self, x):
        # 输入形状: [B, T, 2]

        # 小波降噪
        denoised = self.wavelet_denoise(x)  # [B, T, 2]

        # 计算累积量
        cumulants = self._calc_cumulants(denoised)

        # ResNet特征提取
        x = denoised.permute(0, 2, 1)  # [B, 2, T]
        features = self.resnet(x)  # [B,512,T/16]

        # Transformer处理
        trans_feat = features.permute(2, 0, 1)  # [T/16, B, 512]
        trans_out = self.transformer(trans_feat)  # [T/16, B, 512]
        pooled = torch.mean(trans_out, dim=0)  # [B, 512]

        # 特征融合
        combined = torch.cat([pooled, cumulants], dim=1)
        return self.fc(combined)


# 保持原有残差块定义不变
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


def EnhancedResNetTransformer(num_classes=9):
    return ResNetTransformer(num_classes=num_classes)