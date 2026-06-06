"""
preprocessing.py - 数据清洗与预处理模块
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def check_data_quality(df: pd.DataFrame) -> dict:
    """数据质量检查报告"""
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }
    print("\n=== 数据质量报告 ===")
    print(f"数据集形状: {report['shape']}")
    print(f"缺失值统计:\n{pd.Series(report['missing_values'])}")
    print(f"重复行数: {report['duplicates']}")
    return report


def remove_outliers(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """基于 Z-score 去除异常值"""
    df_clean = df.copy()
    removed = 0
    for col in columns:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        mask = z_scores < threshold
        removed += (~mask).sum()
        df_clean = df_clean[mask]
    print(f"[preprocessing] 异常值移除: {removed} 行, 剩余 {len(df_clean)} 行")
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：构造衍生特征"""
    df = df.copy()
    # 房间/人口比（居住密度）
    df['RoomsPerPerson'] = df['AveRooms'] / (df['AveOccup'] + 1e-6)
    # 卧室占比
    df['BedroomRatio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-6)
    # 收入与房龄交互
    df['IncomeAge'] = df['MedInc'] * df['HouseAge']
    print(f"[preprocessing] 特征工程完成，新增 3 个特征，当前特征数: {df.shape[1]}")
    return df


def preprocess(df: pd.DataFrame, target_col: str = 'MedHouseVal',
               test_size: float = 0.2, random_state: int = 42):
    """
    完整预处理流程
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    # 去异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = [c for c in numeric_cols if c != target_col]
    df_clean = remove_outliers(df, outlier_cols)

    # 特征工程
    df_clean = feature_engineering(df_clean)

    # 分离特征/目标
    feature_cols = [c for c in df_clean.columns if c != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    # 保存处理后数据
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(os.path.join(PROCESSED_DIR, 'cleaned_data.csv'), index=False)
    print(f"[preprocessing] 清洗后数据已保存，训练集: {X_train.shape}, 测试集: {X_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


if __name__ == '__main__':
    from data_loader import load_raw_data
    df = load_raw_data()
    check_data_quality(df)
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print(f"\n特征列: {features}")
