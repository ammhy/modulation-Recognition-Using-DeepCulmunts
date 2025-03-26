import torch
import torch.nn as nn


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
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        return self.relu(x)

class ResNet1DWithTransformer(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv1d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=2
        )

        # 自适应池化和全连接
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 + 2, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 计算高阶累积量
        third, fourth = calculate_cumulants(x)
        cumulants = torch.stack([third, fourth], dim=1)

        # 处理IQ信号
        x = x.permute(0, 2, 1)  # [batch, 2, seq_len]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 512, seq_len]

        # Transformer处理
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # [batch, features, seq_len]

        # 特征提取
        x = self.avgpool(x).squeeze(-1)  # [batch, 512]

        # 特征融合
        combined = torch.cat([x, cumulants], dim=1)  # [batch, 514]
        return self.fc(combined)


def ResNet18WithTransformer(num_classes):
    return ResNet1DWithTransformer(ResidualBlock1D, [2, 2, 2, 2], num_classes)