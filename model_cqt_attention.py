import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
class CQTTransform(nn.Module):
    """常数Q变换模块"""
    def __init__(self, bins=128):
        super().__init__()
        self.bins = bins
        self.n_fft = bins * 2  # 关键修改：确保 bins 与 n_fft 的关系正确

    def forward(self, x):
        # 输入 x 的形状: [batch, channels, seq_len]
        batch, ch, seq = x.shape
        cqt_list = []

        for i in range(ch):
            channel_data = x[:, i, :]
            cqt = torch.stft(
                channel_data,
                n_fft=self.n_fft,  # n_fft = bins * 2
                hop_length=self.bins,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft).to(x.device),
                return_complex=True
            )  # 输出形状 [batch, freq_bins, time]
            cqt = torch.log(torch.abs(cqt) + 1e-6)  # 对数变换
            cqt_list.append(cqt)

        # 合并通道和频率维度
        cqt = torch.stack(cqt_list, dim=1)  # [batch, ch, freq_bins, time]
        cqt = cqt.reshape(batch, ch * (self.n_fft // 2 + 1), -1)  # [batch, ch*(freq_bins), time]
        return cqt

class EnhancedResidualBlock1D(nn.Module):
    """改进的残差块：添加通道注意力机制"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv1d(in_channels, mid_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

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
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.relu(x + residual)

class CrossModalityFusion(nn.Module):
    """跨模态注意力融合模块"""

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.mh_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x1, x2):
        # 输入形状: [batch, channels, seq_len]
        x1 = x1.permute(0, 2, 1)  # [batch, seq_len, channels]
        x2 = x2.permute(0, 2, 1)

        # 注意力机制
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attn_out, _ = self.mh_attn(q, k, v)

        # 残差连接
        return self.layer_norm(attn_out + x1).permute(0, 2, 1)

class ResNet1DWithEnhancedFourier(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()

        # 时域分支
        self.time_conv = nn.Sequential(
            nn.Conv1d(2, 64, 15, 2, 7, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 2),
            *self._build_layers(block, layers, in_channels=64, base_channels=64)
        )

        self.freq_conv = nn.Sequential(
            CQTTransform(bins=128),
            # 输入通道数 = 2 channels * (128*2 // 2 + 1) = 2*129 = 258
            nn.Conv1d(258, 64, 15, 2, 7, bias=False),  # 修正输入通道数为 258
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 2),
            *self._build_layers(block, layers, in_channels=64, base_channels=64)
        )

        # 跨模态融合
        self.fusion = CrossModalityFusion(channels=512, num_heads=8)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _build_layers(self, block, layers, in_channels, base_channels):
        """构建ResNet层"""
        stages = []
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]

        for i in range(4):
            stage = self._make_layer(
                block,
                out_channels=channels[i],
                num_blocks=layers[i],
                in_channels=in_channels,
                stride=strides[i]
            )
            stages.append(stage)
            in_channels = channels[i]
        return stages

    def _make_layer(self, block, out_channels, num_blocks, in_channels, stride=1):
        """构建单个残差层"""
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状: [batch, seq_len, 2]
        x = x.permute(0, 2, 1)  # [batch, 2, seq_len]

        # 时域处理
        x_time = self.time_conv(x)

        # 频域处理
        x_freq = self.freq_conv(x)

        # 跨模态融合
        x_fused = self.fusion(x_time, x_freq)

        # 全局池化
        x_pooled = F.adaptive_avg_pool1d(x_fused, 1).squeeze(-1)

        # 分类
        return self.classifier(x_pooled)

def ResNet34WithEnhancedFourier(num_classes=9):
    return ResNet1DWithEnhancedFourier(EnhancedResidualBlock1D, [3, 4, 6, 3], num_classes)