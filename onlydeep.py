import torch
import torch.nn as nn


# 计算三阶和四阶累积量的函数
def calculate_cumulants(iq_data):
    I = iq_data[:, :, 0]  # 提取实部
    Q = iq_data[:, :, 1]  # 提取虚部

    # 将数据居中，减去均值
    I_centered = I - torch.mean(I, dim=1, keepdim=True)
    Q_centered = Q - torch.mean(Q, dim=1, keepdim=True)

    # 计算三阶累积量 E[(I - mu_I) * (Q - mu_Q)^2]
    third_order = torch.mean(I_centered * Q_centered ** 2, dim=1)

    # 计算四阶累积量 E[(I - mu_I) * (Q - mu_Q)^3]
    fourth_order = torch.mean(I_centered * Q_centered ** 3, dim=1)

    return third_order, fourth_order


# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Linear(input_dim, 64)
        # 第二个全连接层
        self.fc2 = nn.Linear(64, 32)
        # 输出层
        self.fc3 = nn.Linear(32, output_dim)
        # Softmax 层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 计算三阶和四阶累积量
        third_order, fourth_order = calculate_cumulants(x)

        # 将累积量拼接成一个特征向量 (batch_size, 2)，每个信号一个三阶和四阶累积量
        features = torch.stack((third_order, fourth_order), dim=1)

        # 前向传播
        x = torch.relu(self.fc1(features))  # 输入拼接的累积量特征
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# 创建分类器模型的函数
def ModulationClassifier(num_classes, input_size):
    return Classifier(input_dim=2, output_dim=num_classes)  # input_dim=2 因为我们用的是三阶和四阶累积量



