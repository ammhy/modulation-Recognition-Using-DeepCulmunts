import torch
import torch.nn as nn
import torch.optim as optim
from input_val import X_train, y_train, X_test, y_test, X_val, y_val  # 确保标签原始值为 0-8
from model import ResNet18LSTMTransformer
#from onlydeep import ModulationClassifier
#from model_lstm_resnet import ResNet18WithLSTM
#from model_resnet_transformer import ResNet18WithTransformer
#from model_resnet_deep import ResNet34WithCumulants
#from model_fft_transformer import fft_transformer_cumulants
#from model_cqt_attention import ResNet34WithEnhancedFourier
from model_cqt_complex import  EnhancedResNetTransformer
#from model_cqt_without_wave import SimplifiedResNetTransformer
#from model_cqt_complexV1 import ResNet18WithTransformer
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging
# 创建 trainlog 目录
os.makedirs('trainlog', exist_ok=True)
os.makedirs('train_last', exist_ok=True)
# 配置日志记录
logging.basicConfig(filename='trainlog/output_cqtcomplexdenoise0.00001.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

os.makedirs('nine_signal/train', exist_ok=True)


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


# 直接使用 0-8 的标签，无需映射
num_classes = 9  # 9 种信号
input_dim = (2, X_train.shape[1])  # 假设输入为复数信号（实部 + 虚部）
model =  EnhancedResNetTransformer(num_classes=num_classes)
#model = ResNet18WithLSTM(num_classes=num_classes)
#model=fft_transformer_cumulants(num_classes=num_classes)
#model=ResNet34WithCumulants(num_classes=num_classes)
#model = EnhancedResNetTransformer(num_classes=9)
#model=SimplifiedResNetTransformer(num_classes=9)
#model=ResNet18WithTransformer(num_classes=9)
#model=ResNet34WithCumulants(num_classes=9)
# 检查是否有 GPU 可用，如果有则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 打印设备信息
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 200
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
    logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

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
    val_loss /= len(test_loader)
    logging.info(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

    # 判断是否触发早停
    if early_stopper(val_loss):
        logging.info("Early stopping triggered!")
        break

    # 打印前 10 个样本预测结果
    logging.info("\n示例预测结果：")
    for i in range(min(10, len(all_labels))):
        logging.info(f"样本{i + 1}: 正确标签={all_labels[i]}, 预测标签={all_preds[i]}")

    # 计算详细指标
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    logging.info(f"\n整体准确率: {accuracy:.2f}%")

    # 打印混淆矩阵
    labels_list = list(range(num_classes))  # 0-8
    cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
    logging.info("\n混淆矩阵：")
    header = "    " + " ".join([f"Pred {i}" for i in labels_list])
    logging.info(header)
    for i, true_label in enumerate(labels_list):
        logging.info(f"True {true_label} {cm[i]}")

    # 打印分类报告
    logging.info("\n分类报告：")
    target_names = [f'Class {i}' for i in labels_list]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    logging.info(report)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'train_last/output_cqtcomplexdenoise0.00001.pth')
        logging.info('\nBest model saved.')

torch.save(model.state_dict(), 'train_last/output_cqtcomplexdenoise0.00001.pth')
logging.info('\nFinal model saved.')

# 最终测试（只在训练完成后执行一次）
model.load_state_dict(torch.load('train_last/output_cqtcomplexdenoise0.00001.pth'))
model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
# 打印混淆矩阵
labels_list = list(range(num_classes))  # 0-8
cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
logging.info("\n混淆矩阵：")
header = "    " + " ".join([f"Pred {i}" for i in labels_list])
logging.info(header)
for i, true_label in enumerate(labels_list):
     logging.info(f"True {true_label} {cm[i]}")

test_accuracy = 100 * test_correct / len(test_dataset)
logging.info(f"\n最终测试准确率：{test_accuracy:.2f}%")
