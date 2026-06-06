"""
visualization.py - 可视化模块
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 统一配色方案
PALETTE = 'Blues_d'
ACCENT_COLOR = '#1a73e8'


def save_fig(name: str, tight: bool = True):
    """统一保存图片"""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] 已保存: {path}")


def plot_target_distribution(df: pd.DataFrame, target: str = 'MedHouseVal'):
    """目标变量分布（直方图 + KDE）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[target], bins=50, color=ACCENT_COLOR, edgecolor='white', alpha=0.85)
    axes[0].set_title('House Price Distribution', fontsize=13)
    axes[0].set_xlabel('Median House Value (×$100K)')
    axes[0].set_ylabel('Count')

    axes[1].hist(np.log1p(df[target]), bins=50, color='#34a853', edgecolor='white', alpha=0.85)
    axes[1].set_title('Log-transformed Price Distribution', fontsize=13)
    axes[1].set_xlabel('log(1 + Price)')
    axes[1].set_ylabel('Count')

    fig.suptitle('Target Variable Analysis', fontsize=14, fontweight='bold')
    save_fig('01_target_distribution')


def plot_correlation_heatmap(df: pd.DataFrame):
    """相关性热力图"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8}, ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    save_fig('02_correlation_heatmap')


def plot_feature_vs_target(df: pd.DataFrame, target: str = 'MedHouseVal'):
    """关键特征与目标变量散点图"""
    key_features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    colors = [ACCENT_COLOR, '#34a853', '#fbbc04', '#ea4335']
    for i, feat in enumerate(key_features):
        sample = df.sample(min(2000, len(df)), random_state=42)
        axes[i].scatter(sample[feat], sample[target], alpha=0.3, s=8, color=colors[i])
        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel('House Price (×$100K)', fontsize=11)
        axes[i].set_title(f'{feat} vs Price', fontsize=12)

    fig.suptitle('Key Features vs House Price', fontsize=14, fontweight='bold')
    save_fig('03_feature_vs_target')


def plot_geographic_distribution(df: pd.DataFrame, target: str = 'MedHouseVal'):
    """地理热力图（经纬度 + 价格）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sample = df.sample(min(5000, len(df)), random_state=42)
    sc = ax.scatter(
        sample['Longitude'], sample['Latitude'],
        c=sample[target], cmap='YlOrRd',
        alpha=0.6, s=5
    )
    plt.colorbar(sc, ax=ax, label='Median House Value (×$100K)')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('California House Price - Geographic Distribution', fontsize=14, fontweight='bold')
    save_fig('04_geographic_distribution')


def plot_model_comparison(results: dict):
    """模型性能对比柱状图"""
    models = list(results.keys())
    rmse_vals = [results[m]['RMSE'] for m in models]
    r2_vals = [results[m]['R2'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars0 = axes[0].bar(models, rmse_vals, color=[ACCENT_COLOR, '#34a853', '#ea4335', '#fbbc04'][:len(models)], width=0.5)
    axes[0].set_title('RMSE Comparison (lower is better)', fontsize=13)
    axes[0].set_ylabel('RMSE')
    for bar, val in zip(bars0, rmse_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    bars1 = axes[1].bar(models, r2_vals, color=[ACCENT_COLOR, '#34a853', '#ea4335', '#fbbc04'][:len(models)], width=0.5)
    axes[1].set_title('R² Score Comparison (higher is better)', fontsize=13)
    axes[1].set_ylabel('R²')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars1, r2_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    save_fig('05_model_comparison')


def plot_prediction_vs_actual(y_test, y_pred, model_name: str = 'Best Model'):
    """预测值 vs 真实值散点图"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color=ACCENT_COLOR)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Price (×$100K)', fontsize=12)
    ax.set_ylabel('Predicted Price (×$100K)', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    save_fig('06_prediction_vs_actual')


def plot_feature_importance(feature_cols: list, importances: np.ndarray, model_name: str = 'Random Forest'):
    """特征重要性条形图"""
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.plot(kind='barh', ax=ax, color=ACCENT_COLOR, alpha=0.85)
    ax.set_title(f'{model_name} - Feature Importance', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    save_fig('07_feature_importance')
