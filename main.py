"""
main.py - 房价预测分析主程序
运行: python main.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_loader import load_raw_data, get_feature_descriptions
from preprocessing import check_data_quality, preprocess
from visualization import (
    plot_target_distribution, plot_correlation_heatmap,
    plot_feature_vs_target, plot_geographic_distribution,
    plot_model_comparison, plot_prediction_vs_actual,
    plot_feature_importance
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """训练多个模型并对比"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n[training] 训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': model}
        predictions[name] = y_pred
        print(f"  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")

    return results, predictions


def save_report(results: dict, feature_cols: list):
    """保存分析报告到文本文件"""
    report_path = os.path.join(REPORTS_DIR, 'model_performance.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  加州房价预测 - 模型性能报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"使用特征: {feature_cols}\n\n")
        f.write(f"{'模型':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}\n")
        f.write("-" * 60 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<25} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f} {metrics['R2']:>10.4f}\n")
        f.write("\n最优模型（R²最高）: ")
        best = max(results, key=lambda k: results[k]['R2'])
        f.write(f"{best}  R²={results[best]['R2']:.4f}\n")
    print(f"\n[report] 报告已保存: {report_path}")
    return best


def main():
    print("=" * 60)
    print("  加州房价预测分析 - 完整流程")
    print("=" * 60)

    # 1. 数据加载
    print("\n[Step 1] 加载数据...")
    df = load_raw_data()

    # 2. EDA 可视化
    print("\n[Step 2] 探索性数据分析...")
    check_data_quality(df)
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)
    plot_geographic_distribution(df)

    # 3. 预处理
    print("\n[Step 3] 数据预处理...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)

    # 4. 训练与评估
    print("\n[Step 4] 模型训练与评估...")
    results, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. 可视化结果
    print("\n[Step 5] 生成结果可视化...")
    plot_model_comparison(results)

    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = results[best_model_name]['model']
    plot_prediction_vs_actual(y_test.values, predictions[best_model_name], best_model_name)

    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(feature_cols, best_model.feature_importances_, best_model_name)

    # 6. 保存报告
    print("\n[Step 6] 保存报告...")
    save_report(results, feature_cols)

    print("\n" + "=" * 60)
    print(f"  分析完成！最优模型: {best_model_name}")
    print(f"  R² = {results[best_model_name]['R2']:.4f}")
    print(f"  RMSE = {results[best_model_name]['RMSE']:.4f}")
    print("  图表已保存至 outputs/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
