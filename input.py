import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# 加载 .npy 文件
data = np.load('E:\\DeepCumulants\\combinedDataset.npy', allow_pickle=True).item()

# 提取数据
ID = data['ID']
SNR = data['SNR']
labels = data['labels']
features = data['features']


label_condition = np.isin(labels, [0,1,2,3,4,5,6,7,8])

filtered_indices = np.where(label_condition)[0]


# 筛选出符合条件的数据
filtered_ID = ID[filtered_indices]
filtered_SNR = SNR[filtered_indices]
filtered_labels = labels[filtered_indices]
filtered_features = features[filtered_indices]

# 从 filtered_features 中提取 I 和 Q 分量
filtered_I_component = filtered_features[:, :1024]   # 前 1024 列为 I 分量
filtered_Q_component = filtered_features[:, 1024:]   # 后 1024 列为 Q 分量

# 将 I 和 Q 分量合并为单个特征矩阵 (样本数, 序列长度, 2)
filtered_IQ_samples = np.stack((filtered_I_component, filtered_Q_component), axis=-1)

# 数据归一化
filtered_IQ_samples = filtered_IQ_samples / np.max(np.abs(filtered_IQ_samples))


# 原始完整数据
X_full = filtered_features
y_full = filtered_labels
SNR_full = filtered_SNR

# 首次分割：训练+临时集（80%训练，20%临时）
X_train, X_temp, y_train, y_temp, SNR_train, SNR_temp = train_test_split(
    X_full, y_full, SNR_full,
    test_size=0.2,
    stratify=y_full,  # 保持类别分布
    random_state=42
)

# 二次分割：验证+测试集（各占 10%）
X_val, X_test, y_val, y_test, SNR_val, SNR_test = train_test_split(
    X_temp, y_temp, SNR_temp,
    test_size=0.5,
    stratify=y_temp,  # 保持类别分布
    random_state=42
)

# 新增验证集 Tensor 转换
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
SNR_val = torch.tensor(SNR_val, dtype=torch.float32)

# 训练集和测试集的 Tensor 转换
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
SNR_train = torch.tensor(SNR_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
SNR_test = torch.tensor(SNR_test, dtype=torch.float32)

# 统计每个集合中每种 label 的数量
def count_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, counts))
    return label_count_dict


train_label_count = count_labels(y_train)
val_label_count = count_labels(y_val)
test_label_count = count_labels(y_test)

# 输出结果
print("训练集中每种 label 的数量：")
for label, count in train_label_count.items():
    print(f"Label: {label}, Count: {count}")

print("\n验证集中每种 label 的数量：")
for label, count in val_label_count.items():
    print(f"Label: {label}, Count: {count}")

print("\n测试集中每种 label 的数量：")
for label, count in test_label_count.items():
    print(f"Label: {label}, Count: {count}")

'''
# 划分数据集 (训练集和测试集)
X_train, X_test, y_train, y_test, SNR_train, SNR_test = train_test_split(
    filtered_IQ_samples, filtered_labels, filtered_SNR, test_size=0.2, random_state=42
)

# 查看所有样本中每个标签的数量
print("\nLabel distribution in filtered samples:")
unique_labels, counts = np.unique(filtered_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")

# 查看训练集中每个标签的数量
print("\nLabel distribution in the training set:")
unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_train_labels, train_counts):
    print(f"Label: {label}, Count: {count}")
sys.stdout.flush()  # 强制刷新输出，实时显示训练过程


# 将数据转换为 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
SNR_test = torch.tensor(SNR_test, dtype=torch.float32)
'''