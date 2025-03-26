import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from input_val import X_train, y_train, X_test, y_test, X_val, y_val
from model_cqt_complex import EnhancedResNetTransformer  # 请确保模型类导入路径正确
from sklearn.metrics import confusion_matrix, classification_report
import logging
import sys

# 配置日志记录
logging.basicConfig(filename='test_simple_signal.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
def load_model(model_path):
    num_classes = 9
    model = EnhancedResNetTransformer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# 预处理测试数据
def preprocess_test_data(data_path):
    # 加载 .npy 文件
    data = np.load(data_path, allow_pickle=True).item()

    # 提取数据
    ID = data['ID']
    SNR = data['SNR']
    labels = data['labels']
    features = data['features']

    # 筛选符合条件的数据
    label_condition = np.isin(labels, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    filtered_indices = np.where(label_condition)[0]

    # 筛选后的数据
    filtered_labels = labels[filtered_indices]
    filtered_features = features[filtered_indices]

    # 提取 I 和 Q 分量
    filtered_I_component = filtered_features[:, :1024]
    filtered_Q_component = filtered_features[:, 1024:]

    # 合并为 IQ 样本
    filtered_IQ_samples = np.stack((filtered_I_component, filtered_Q_component), axis=-1)

    # 归一化
    filtered_IQ_samples = filtered_IQ_samples / np.max(np.abs(filtered_IQ_samples))

    # 转换为 PyTorch Tensor
    X_test = torch.tensor(filtered_IQ_samples, dtype=torch.float32).to(device)
    y_test = torch.tensor(filtered_labels, dtype=torch.long).to(device)

    return X_test, y_test

# 测试模型性能
def test_model(model, X_test, y_test):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_test).sum().item()
        total += y_test.size(0)

        # 收集预测结果
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

    accuracy = 100 * correct / total
    logging.info(f"测试集准确率: {accuracy:.2f}%")

    # 打印混淆矩阵
    logging.info("\n混淆矩阵：")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(9)))
    logging.info(cm)

    # 打印分类报告
    logging.info("\n分类报告：")
    report = classification_report(all_labels, all_preds, labels=list(range(9)), target_names=[f'Class {i}' for i in range(9)])
    logging.info(report)

# 主函数
def main():
    # 参数配置
    model_path = 'E:\\deepCumulants\\train_last\\output_cqt_complex0.00005.pth'  # 替换为您的模型路径
    data_path = 'trainDataset.npy'  # 替换为您的测试数据路径

    # 加载模型
    model = load_model(model_path)

    # 预处理测试数据
    X_test, y_test = preprocess_test_data(data_path)

    # 测试模型
    test_model(model, X_test, y_test)

if __name__ == "__main__":
    main()