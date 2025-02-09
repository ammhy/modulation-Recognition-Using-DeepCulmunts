import torch
import torch.nn as nn
import torch.optim as optim
from input import X_train, y_train, X_test, y_test  # 导入训练和测试数据
from onlydeep import ModulationClassifier
import os
import sys
os.makedirs('train', exist_ok=True)

# 获取输出类别的数量
num_classes = 9  # 数据集中有 9 种不同的调制方式

# 动态计算全连接层输入大小
input_dim = (2, X_train.shape[1])


# 假设 num_classes 和 input_dim 已定义
model = ModulationClassifier(num_classes=num_classes, input_size=input_dim)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练模型
num_epochs = 200
batch_size = 128

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_accuracy = 0.0  # 记录最佳模型的准确率

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

    # 调整学习率
    scheduler.step()

    # 打印每个 epoch 的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 测试模型在测试集上的准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
    sys.stdout.flush()  # 强制刷新输出，实时显示训练过程

    # 保存测试集上表现最好的模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'train/best_modulation_classifier.pth')
        print('Best model saved.')

print('Training complete.')

# 创建 'train' 文件夹（如果不存在的话）
os.makedirs('train', exist_ok=True)

# 保存最终模型权重
torch.save(model.state_dict(), 'train/modulation_classifier.pth')
print('Model saved successfully in train directory.')
