import torch
import torch.nn as nn


def calculate_cumulants(iq_data):
    I = iq_data[:, :, 0]
    Q = iq_data[:, :, 1]
    I_centered = I - torch.mean(I, dim=1, keepdim=True)
    Q_centered = Q - torch.mean(Q, dim=1, keepdim=True)
    third_order = torch.mean(I_centered * Q_centered ** 2, dim=1)
    fourth_order = torch.mean(I_centered * Q_centered ** 3, dim=1)
    return third_order, fourth_order


class ResidualBlock1D(nn.Module):
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


class ResNet1DWithLSTM(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv1d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=0.3)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(512, 512, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)

        # Feature fusion
        self.cumulant_fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )
        self.fc = nn.Linear(512 * 2 + 512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout=0.2):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Cumulants
        third, fourth = calculate_cumulants(x)
        cumulants = self.cumulant_fc(torch.stack([third, fourth], dim=1))

        # ResNet processing
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # LSTM processing
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = torch.cat([x[:, -1, :512], x[:, 0, 512:]], dim=1)  # Bidirectional

        # Fusion
        combined = torch.cat([x, cumulants], dim=1)
        return self.fc(combined)


def ResNet18WithLSTM(num_classes):
    return ResNet1DWithLSTM(ResidualBlock1D, [2, 2, 2, 2], num_classes)