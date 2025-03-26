import torch
import torch.nn as nn

# ResNet18 结合 Transformer 模块
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        # 残差块中的第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 残差块中的第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果需要匹配维度，使用捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 通过第一个卷积层、批归一化和 ReLU 激活函数
        out = self.relu(self.bn1(self.conv1(x)))
        # 通过第二个卷积层和批归一化
        out = self.bn2(self.conv2(out))
        # 加上捷径连接
        out += self.shortcut(x)
        # 应用 ReLU 激活函数
        out = self.relu(out)
        return out


# 加入 Transformer 和 LSTM 的 ResNet
class ResNet1DWithTransformer(nn.Module):
    def __init__(self, block, layers, num_classes, input_size, lstm_input_size=2, lstm_hidden_size=128,
                 lstm_num_layers=2, num_heads=6, num_transformer_layers=3, d_model=384, dropout_rate=0.4):
        super(ResNet1DWithTransformer, self).__init__()
        self.in_channels = 64

        # LSTM 层（添加在最开始）
        self.lstm_before_resnet = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                          num_layers=lstm_num_layers, batch_first=True, dropout=dropout_rate)

        # 初始卷积层
        self.conv1 = nn.Conv1d(lstm_hidden_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, d_model, layers[3], stride=2)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Dropout 层用于正则化
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层用于最终分类
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # 第一个残差块，可能需要改变输入输出通道或步幅
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 其余的残差块，保持输入输出通道一致
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # LSTM 前向传播 (batch_size, seq_len, 2) -> (batch_size, seq_len, lstm_hidden_size)
        x, _ = self.lstm_before_resnet(x)

        # 转换形状以适应卷积层 (batch_size, seq_len, lstm_hidden_size) -> (batch_size, lstm_hidden_size, seq_len)
        x = x.permute(0, 2, 1)

        # 通过初始卷积层、批归一化和 ReLU 激活函数
        x = self.relu(self.bn1(self.conv1(x)))
        # 最大池化层
        x = self.maxpool(x)
        # 通过残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 为 Transformer 做准备：(batch_size, d_model, seq_len) -> (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1)
        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # 输出形状: (batch_size, seq_len, d_model)

        # 取最后一个时间步的输出
        x = x[:, -1, :]  # 形状: (batch_size, d_model)

        # 应用 Dropout
        x = self.dropout(x)

        # 通过全连接层进行最终分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def ResNet18WithTransformer(num_classes, input_size):
    return ResNet1DWithTransformer(ResidualBlock1D, [2, 2, 2, 2], num_classes=num_classes, input_size=input_size)

# 实例化模型
# num_classes = 9  # 数据集中有 9 种不同的调制方式
# input_size = (2, X_train.shape[1])
# model = ResNet18WithTransformer(num_classes=num_classes, input_size=input_size)
# print(model)
