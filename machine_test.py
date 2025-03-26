import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import load
from scipy import stats


# --------------------- 数据加载 ---------------------
def load_new_data(file_path):
    """加载新数据集"""
    data = np.load(file_path, allow_pickle=True).item()
    signals = data['features']
    # 分离I/Q分量并转换为复数信号
    complex_signals = signals[:, :1024] + 1j * signals[:, 1024:]
    return complex_signals, data['labels'], data['SNR']


# --------------------- 特征工程 ---------------------
def calculate_hoc(sig):
    """计算四阶累积量特征"""
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    return np.mean(sig ** 4) - 3 * (np.mean(sig ** 2)) ** 2


def extract_features(signal, block_size=256):
    """特征提取函数"""
    features = []
    num_blocks = len(signal) // block_size

    for i in range(num_blocks):
        block = signal[i * block_size: (i + 1) * block_size]
        real_part = block.real
        imag_part = block.imag

        # 时域统计特征
        features.extend([
            np.abs(block).mean(),  # 幅度均值
            np.abs(block).std(),  # 幅度标准差
            np.angle(block).var(),  # 相位方差
            stats.skew(real_part),  # 实部偏度
            stats.kurtosis(imag_part),  # 虚部峰度
            calculate_hoc(real_part),  # 实部四阶累积量
            calculate_hoc(imag_part),  # 虚部四阶累积量
            np.max(np.abs(np.fft.fft(real_part)))  # 实部FFT峰值
        ])

    return np.array(features)


# --------------------- 主流程 ---------------------
def validate_on_new_data():
    # 1. 加载新数据集
    new_complex_signals, new_labels, new_snr = load_new_data('trainDataset.npy')

    # 2. 提取特征
    print("开始提取新数据集特征...")
    X_new = np.array([extract_features(sig) for sig in new_complex_signals])
    y_new = np.array(new_labels)

    # 3. 加载之前保存的模型和预处理流水线
    print("加载模型和预处理流水线...")
    model = load('modulation_classifierold0_1.joblib')
    pipeline = load('preprocessing_pipelineold0_01.joblib')

    # 4. 对新数据进行预处理
    X_new_processed = pipeline.named_steps['scaler'].transform(X_new)  # 仅使用scaler

    # 5. 使用模型进行预测
    print("开始预测...")
    y_pred = model.predict(X_new_processed)

    # 6. 评估模型性能
    accuracy = accuracy_score(y_new, y_pred)
    print("\n新数据集上的正确率（Accuracy）: {:.4f}".format(accuracy))

    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(y_new, y_pred, digits=4))

    # 混淆矩阵可视化
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_new, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,  # 替换为实际类别名称
                yticklabels=CLASS_NAMES)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('新数据集混淆矩阵')
    plt.show()


if __name__ == "__main__":
    # 定义类别名称（根据实际标签顺序）
    CLASS_NAMES = [
        'OFDM', '2FSK', '4FSK', '8FSK',
        'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'
    ]
    validate_on_new_data()