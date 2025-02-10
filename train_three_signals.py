import torch
import torch.nn as nn
import torch.optim as optim
from input import X_train, y_train, X_test, y_test  # 确保标签原始值为3/4/5
from model import ResNet18WithTransformer
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs('train', exist_ok=True)

# ================= 新增配置 =================
label_map = {3: 0, 4: 1, 5: 2}  # 将原始标签映射到0-2
reverse_map = {v: k for k, v in label_map.items()}  # 用于结果反转换

# 转换标签为0-2
y_train = torch.tensor([label_map[int(x)] for x in y_train])
y_test = torch.tensor([label_map[int(x)] for x in y_test])
# ===========================================

num_classes = 3  # 修改为3个类别
input_dim = (2, X_train.shape[1])  # 假设输入为复数信号（实部+虚部）

model = ResNet18WithTransformer(num_classes=num_classes, input_size=input_dim)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 200
batch_size = 128

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_accuracy = 0.0

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

    # ================= 新增预测可视化 =================
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为原始标签
    original_preds = [reverse_map[p] for p in all_preds]
    original_labels = [reverse_map[l] for l in all_labels]

    # 打印前10个样本预测结果
    print("\n示例预测结果：")
    for i in range(min(10, len(original_labels))):
        print(f"样本{i + 1}: 正确标签={original_labels[i]}, 预测标签={original_preds[i]}")

    # 计算详细指标
    accuracy = 100 * np.mean(np.array(original_preds) == np.array(original_labels))
    print(f"\n整体准确率: {accuracy:.2f}%")

    # 打印混淆矩阵
    cm = confusion_matrix(original_labels, original_preds, labels=[3, 4, 5])
    print("\n混淆矩阵（原始标签）：")
    print("     Pred 3  Pred 4  Pred 5")
    for i, true_label in enumerate([3, 4, 5]):
        print(f"True {true_label} {cm[i]}")

    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(original_labels, original_preds,
                                target_names=['Class 3', 'Class 4', 'Class 5']))
    # ================================================

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'train/best_modulation_classifier.pth')
        print('\nBest model saved.')

    sys.stdout.flush()

# 保存最终模型
torch.save(model.state_dict(), 'train/modulation_classifier.pth')
print('\nFinal model saved.')