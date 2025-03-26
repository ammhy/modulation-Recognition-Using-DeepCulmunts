import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from scipy import stats


# --------------------- 数据加载 ---------------------
def load_data(file_path):
    """加载并处理原始数据"""
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
    """并行特征提取函数"""
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
def main():
    # 1. 加载数据
    complex_signals, labels, snr = load_data('data.npy')

    # 2. 并行特征提取（使用全部CPU核心）
    print("开始特征提取...")
    X = np.array(Parallel(n_jobs=-1)(
        delayed(extract_features)(sig) for sig in complex_signals
    ))
    y = np.array(labels)

    # 3. 数据预处理
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
    ])
    X_processed, y_processed = pipeline.fit_resample(X, y)

    # 4. 分层划分数据集（保持SNR分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=0.3,
        stratify=y_processed,
        random_state=42
    )

    # 5. 构建并训练XGBoost模型
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        n_jobs=-1,
        random_state=42
    )

    print("开始训练模型...")
    model.fit(X_train, y_train)

    # 6. 评估模型
    y_pred = model.predict(X_test)

    # 计算正确率
    accuracy = accuracy_score(y_test, y_pred)
    print("\n正确率（Accuracy）: {:.4f}".format(accuracy))

    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, digits=4))

    # 混淆矩阵可视化
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,  # 替换为实际类别名称
                yticklabels=CLASS_NAMES)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    # 7. 保存模型

    import joblib
    joblib.dump(model, 'modulation_classifierold0_1.joblib')
    joblib.dump(pipeline, 'preprocessing_pipelineold0_01.joblib')


if __name__ == "__main__":
    # 定义类别名称（根据实际标签顺序）
    CLASS_NAMES = [
        'OFDM', '2FSK', '4FSK', '8FSK',
        'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'
    ]
    main()