
#OFDM, 2FSK, 4FSK, 8FSK, BPSK, QPSK, 8PSK, 16QAM, 64QAM.

import torch
import torch.nn as nn
import torch.optim as optim
from input_val import X_train, y_train, X_test, y_test ,X_val,y_val # 确保标签原始值为你需要的六个类别
from model import ResNet18WithTransformer
from onlydeep import ModulationClassifier
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs('three_signal/train', exist_ok=True)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        if mode == 'min':
            self.best_score = float('inf')
            self._compare = lambda x, y: x < y - min_delta
        elif mode == 'max':
            self.best_score = float('-inf')
            self._compare = lambda x, y: x > y + min_delta
        else:
            raise ValueError("mode must be either 'min' or 'max'")

    def __call__(self, score):
        if self._compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



# ================= 新增配置 =================
#假设原始标签为 1, 2, 3, 4, 5, 6，将其映射到 0 - 5
label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
reverse_map = {v: k for k, v in label_map.items()}  # 用于结果反转换

# 转换标签为 0 - 5
y_train = torch.tensor([label_map[int(x)] for x in y_train])
y_test = torch.tensor([label_map[int(x)] for x in y_test])
y_val = torch.tensor([label_map[int(x)] for x in y_val])
# ===========================================
# 直接使用原始标签
#y_train = torch.tensor(y_train)  # 使用原始标签
#y_test = torch.tensor(y_test)    # 使用原始标签

num_classes = 6  # 类别数量修改为 6
input_dim = (2, X_train.shape[1])  # 假设输入为复数信号（实部 + 虚部）

model = ModulationClassifier(num_classes=num_classes, input_size=input_dim)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 100
batch_size = 128

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建验证集DataLoader
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_accuracy = 0.0
early_stopper = EarlyStopper(patience=10, min_delta=0.001, mode='min')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 打印验证集损失
    val_loss /= len(val_loader)
    print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

    # 判断是否触发早停
    if early_stopper(val_loss):
        print("Early stopping triggered!")
        break

    # 转换为原始标签
    original_preds = [reverse_map[p] for p in all_preds]
    original_labels = [reverse_map[l] for l in all_labels]

    # 打印前 10 个样本预测结果
    print("\n示例预测结果：")
    for i in range(min(10, len(original_labels))):
        print(f"样本{i + 1}: 正确标签={original_labels[i]}, 预测标签={original_preds[i]}")

    # 计算详细指标
    accuracy = 100 * np.mean(np.array(original_preds) == np.array(original_labels))
    print(f"\n整体准确率: {accuracy:.2f}%")

    # 打印混淆矩阵
    labels_list = [1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(original_labels, original_preds, labels=labels_list)
    print("\n混淆矩阵（原始标签）：")
    header = "    " + " ".join([f"Pred {i}" for i in labels_list])
    print(header)
    for i, true_label in enumerate(labels_list):
        print(f"True {true_label} {cm[i]}")

    # 打印分类报告
    print("\n分类报告：")
    target_names = [f'Class {i}' for i in labels_list]
    print(classification_report(original_labels, original_preds,
                                target_names=target_names))
    # ================================================
    # 保存最佳模型,基于val
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'three_signal/train/best_modulation_classifier.pth')
        print('\nBest model saved.')

    sys.stdout.flush()

 # 保存最终模型
torch.save(model.state_dict(), 'train/modulation_classifier.pth')
print('\nFinal model saved.')


# 最终测试（只在训练完成后执行一次）
model.load_state_dict(torch.load('three_signal/train/best_modulation_classifier.pth'))  # 修改路径一致性
model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()

test_accuracy = 100 * test_correct / len(test_dataset)
print(f"\n最终测试准确率：{test_accuracy:.2f}%")