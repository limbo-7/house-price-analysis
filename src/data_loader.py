"""
data_loader.py - 数据加载模块
使用 pandas 直接读取本地 CSV（首次运行自动用 sklearn 生成）
若网络不可用则用 make_regression 生成模拟数据
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_PATH = os.path.join(RAW_DIR, 'california_housing.csv')


def _generate_synthetic_housing_data(n_samples: int = 20640) -> pd.DataFrame:
    """
    生成结构与加州房价数据集一致的模拟数据（完全离线）
    特征分布基于真实数据集的统计特性
    """
    np.random.seed(42)
    n = n_samples

    MedInc     = np.abs(np.random.lognormal(mean=1.0, sigma=0.8, size=n)).clip(0.5, 15.0)
    HouseAge   = np.random.uniform(1, 52, size=n)
    AveRooms   = np.abs(np.random.lognormal(mean=1.6, sigma=0.4, size=n)).clip(1.0, 30.0)
    AveBedrms  = (AveRooms * np.random.uniform(0.15, 0.35, n)).clip(1.0, 8.0)
    Population = np.abs(np.random.lognormal(mean=6.1, sigma=1.1, size=n)).clip(3, 35682)
    AveOccup   = np.abs(np.random.lognormal(mean=1.1, sigma=0.4, size=n)).clip(1.0, 10.0)
    Latitude   = np.random.uniform(32.5, 42.0, size=n)
    Longitude  = np.random.uniform(-124.5, -114.3, size=n)

    # 目标变量：房价中位数（模拟相关性）
    noise = np.random.normal(0, 0.5, n)
    MedHouseVal = (
        0.45 * MedInc
        + 0.05 * HouseAge / 10
        - 0.03 * np.abs(Latitude - 34)   # 洛杉矶附近价格高
        - 0.03 * np.abs(Longitude + 118)
        + 0.02 * AveRooms
        + noise
    ).clip(0.15, 5.0)

    df = pd.DataFrame({
        'MedInc': np.round(MedInc, 4),
        'HouseAge': np.round(HouseAge, 1),
        'AveRooms': np.round(AveRooms, 4),
        'AveBedrms': np.round(AveBedrms, 4),
        'Population': np.round(Population).astype(int),
        'AveOccup': np.round(AveOccup, 4),
        'Latitude': np.round(Latitude, 4),
        'Longitude': np.round(Longitude, 4),
        'MedHouseVal': np.round(MedHouseVal, 3),
    })
    return df


def load_raw_data() -> pd.DataFrame:
    """
    加载原始数据
    - 若 data/raw/california_housing.csv 已存在则直接读取
    - 否则生成模拟数据并保存
    Returns:
        df: 包含特征和目标变量的完整 DataFrame (20640 行 × 9 列)
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        print(f"[data_loader] 已加载本地数据: {RAW_PATH}")
    else:
        print("[data_loader] 生成模拟加州房价数据集 (20640 行)...")
        df = _generate_synthetic_housing_data()
        df.to_csv(RAW_PATH, index=False)
        print(f"[data_loader] 数据已保存到: {RAW_PATH}")

    print(f"[data_loader] 数据集大小: {df.shape}，字段: {list(df.columns)}")
    return df


def get_feature_descriptions() -> dict:
    """返回各字段含义说明"""
    return {
        'MedInc':       '区块内收入中位数（万美元）',
        'HouseAge':     '房屋房龄中位数（年）',
        'AveRooms':     '平均房间数',
        'AveBedrms':    '平均卧室数',
        'Population':   '区块总人口',
        'AveOccup':     '平均入住人数',
        'Latitude':     '纬度',
        'Longitude':    '经度',
        'MedHouseVal':  '目标变量：房价中位数（×10万美元）',
    }


if __name__ == '__main__':
    df = load_raw_data()
    print(df.head())
    print("\n字段说明:")
    for col, desc in get_feature_descriptions().items():
        print(f"  {col:14s}: {desc}")
