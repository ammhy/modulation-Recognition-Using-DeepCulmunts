import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from torch.nn import TransformerEncoder


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def calculate_cumulants(iq_data):
    I = iq_data[:, :, 0]
    Q = iq_data[:, :, 1]

    # Center the data
    I_centered = I - torch.mean(I, dim=1, keepdim=True)
    Q_centered = Q - torch.mean(Q, dim=1, keepdim=True)

    # 计算三阶和四阶累积量
    third_order = torch.mean(I_centered * Q_centered ** 2, dim=1)
    fourth_order = torch.mean(I_centered * Q_centered ** 3, dim=1)
    return third_order, fourth_order


class WaveletTransform(nn.Module):
    """小波变换特征提取"""

    def __init__(self, n_filters=4):
        super().__init__()
        self.conv = nn.Conv1d(2, n_filters, 32, padding=15, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return F.relu(self.conv(x))


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1)  # 输出形状调整为 (b, c, 1)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out


class EnhancedResidualBlock1D(nn.Module):
    """改进的残差块：添加空间注意力机制"""

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

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
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

        x = self.ca(x) * x  # 通道注意力
        x = self.sa(x) * x  # 空间注意力
        return self.relu(x + residual)


class CQTTransform(nn.Module):
    """常数Q变换模块"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 输入 x 的形状: [batch, channels, seq_len]
        batch_size, channels, seq_len = x.shape

        # 对每个通道分别进行 STFT 变换
        cqt_list = []
        for i in range(channels):
            # 提取单个通道的数据 [batch, seq_len]
            channel_data = x[:, i, :]

            # 计算 STFT
            cqt = torch.stft(
                channel_data,
                n_fft=512,
                hop_length=128,
                win_length=512,
                window=torch.hann_window(512).to(x.device),  # 使用汉宁窗
                return_complex=True
            )
            # 取绝对值并进行对数变换
            cqt = torch.log(torch.abs(cqt) + 1e-6)  # 避免 log(0)
            cqt_list.append(cqt)

        # 将多个通道的结果拼接在一起
        cqt = torch.stack(cqt_list, dim=1)  # [batch, channels, freq_bins, time]

        # 调整维度顺序为 [batch, freq_bins, time, channels]
        cqt = cqt.permute(0, 2, 3, 1)

        # 将 channels 维度展平
        cqt = cqt.reshape(batch_size, -1, cqt.size(2))  # [batch, freq_bins * channels, time]

        return cqt


class ResNet1DWithEnhancedFourier(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()

        # 新增小波变换分支
        self.wavelet = nn.Sequential(
            WaveletTransform(n_filters=4),
            nn.MaxPool1d(4)
        )

        # 时域分支
        self.time_conv = nn.Sequential(
            nn.Conv1d(2, 64, 15, 2, 7, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 2),
            *self._build_layers(block, layers, in_channels=64, base_channels=64)
        )

        # 频域分支（使用 CQTTransform 模块）
        self.freq_conv = nn.Sequential(
            CQTTransform(),  # 使用独立的 Module
            nn.Conv1d(257 * 2, 64, 15, 2, 7, bias=False),  # 输入通道数为 257 * 2 = 514
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 2),
            *self._build_layers(block, layers, in_channels=64, base_channels=64)
        )

        # 改进的跨模态融合模块
        self.fusion = CrossModalityFusion(
            channels=512,
            num_heads=8,
            dropout=0.2
        )

        # 时序增强（使用更深的Transformer）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_enhance = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        # 高阶统计量增强
        self.cumulants_proj = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 512)
        )

        # 分类器（添加多尺度特征）
        self.classifier = nn.Sequential(
            nn.Linear(1028, 1024),  # 输入维度调整为 1028
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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
        # 小波特征
        wavelet_feat = self.wavelet(x.permute(0, 2, 1))
        wavelet_feat = F.adaptive_avg_pool1d(wavelet_feat, 1).squeeze(-1)

        # 高阶统计量
        third, fourth = calculate_cumulants(x)
        cumulants = self.cumulants_proj(torch.stack([third, fourth], dim=1))
        ####

        # 时域处理
        x_time = self.time_conv(x.permute(0, 2, 1))

        # 频域处理
        x_freq = self.freq_conv(x.permute(0, 2, 1))

        # 跨模态融合
        fused = self.fusion(x_time, x_freq)

        # 时序增强
        temporal = self.temporal_enhance(fused.permute(0, 2, 1))
        temporal = F.adaptive_avg_pool1d(temporal.permute(0, 2, 1), 1).squeeze(-1)

        # 特征聚合
        combined = torch.cat([temporal, cumulants, wavelet_feat], dim=1)
        return self.classifier(combined)


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
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn_out, _ = self.mh_attn(q, k, v)
        return self.layer_norm(attn_out + x1).permute(0, 2, 1)


def fft_transformer_cumulants(num_classes=9):
    return ResNet1DWithEnhancedFourier(EnhancedResidualBlock1D, [3, 4, 6, 3], num_classes)