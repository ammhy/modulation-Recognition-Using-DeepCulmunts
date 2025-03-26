import torch
import torch.nn as nn


def calculate_cumulants(iq_data):
    """
    计算 IQ 数据的三阶和四阶累积量
    """
    I = iq_data[:, :, 0]
    Q = iq_data[:, :, 1]
    I_centered = I - torch.mean(I, dim=1, keepdim=True)
    Q_centered = Q - torch.mean(Q, dim=1, keepdim=True)
    third_order = torch.mean(I_centered * Q_centered ** 2, dim=1)
    fourth_order = torch.mean(I_centered * Q_centered ** 3, dim=1)
    return third_order, fourth_order


class ResidualBlock1D(nn.Module):
    """
    1D 残差块
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.dropout(self.relu(out))


class ResNet1DWithCumulants(nn.Module):
    """
    结合累积量的 1D ResNet 模型
    """
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64

        # 初始层
        self.conv1 = nn.Conv1d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        # 深层残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4])

        # 多尺度池化
        self.pyramid_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(16),  # 调整为较小的输出尺寸
            nn.AdaptiveAvgPool1d(8),
            nn.AdaptiveAvgPool1d(4)
        ])

        # 累积量处理网络
        self.cumulant_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512))

        # 动态计算全连接层输入维度
        self.fc_input_dim = None
        self.fc = None

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建残差层
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        """
        # 计算累积量
        third, fourth = calculate_cumulants(x)
        cumulants = self.cumulant_net(torch.stack([third, fourth], dim=1))

        # 深层特征提取
        x = x.permute(0, 2, 1)  # 调整输入形状为 (batch_size, channels, sequence_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # 多尺度池化
        pyramid_features = []
        for pool in self.pyramid_pool:
            pyramid_features.append(pool(x5).flatten(1))
        pyramid_features = torch.cat(pyramid_features, dim=1)

        # 特征融合
        combined = torch.cat([pyramid_features, cumulants], dim=1)

        # 动态初始化全连接层
        if self.fc is None:
            self.fc_input_dim = combined.size(1)
            self.fc = nn.Linear(self.fc_input_dim, 9).to(x.device)

        return self.fc(combined)


def ResNet34WithCumulants(num_classes):
    """
    构建 ResNet-34 结合累积量的模型
    """
    return ResNet1DWithCumulants(ResidualBlock1D, [3, 4, 6, 3, 2], num_classes)