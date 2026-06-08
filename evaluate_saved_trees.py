"""
evaluate_saved_trees.py
========================
Kịch bản tự động tải các mô hình LightGBM và XGBoost đã được huấn luyện sẵn,
sau đó đánh giá các chỉ số MAE, RMSE, và R2 trên tập Test Hà Nội (timestamp >= 2025-10-01).
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import FeatureEngineer

# Fix Unicode cho Windows Console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_COLUMNS = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
TARGET_DISPLAY_NAMES = {
    "pm25_obs": "PM2.5",
    "pm10_obs": "PM10",
    "no2_pseudo": "NO₂",
    "so2_pseudo": "SO₂",
    "co_pseudo": "CO",
    "o3_pseudo": "O₃",
}

def load_and_combine_datasets():
    hn_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hanoi.parquet"
    hcm_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hcmc.parquet"
    
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "../dataset_hanoi.parquet"
        hcm_path = "../dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "dataset_hanoi.parquet"
        hcm_path = "dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        raise FileNotFoundError("❌ Thiếu file dataset_hanoi.parquet hoặc dataset_hcmc.parquet.")
        
    print(f"📖 Đang đọc dữ liệu Hà Nội...")
    df_hn = pd.read_parquet(hn_path)
    df_hn["city"] = "hanoi"
    
    print(f"📖 Đang đọc dữ liệu TP.HCM...")
    df_hcm = pd.read_parquet(hcm_path)
    df_hcm["city"] = "hcmc"
    
    df_combined = pd.concat([df_hn, df_hcm], ignore_index=True)
    df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    
    # LỌC DỮ LIỆU TẬP TEST: Chỉ cần giữ lại từ 2025-09-23 00:00:00 trở đi (2025-10-01 trừ đi 8 ngày lag)
    # Giúp giảm dữ liệu từ 2.5 triệu dòng xuống ~200k dòng, giải phóng 90% RAM và chống lỗi OOM.
    test_start_date = pd.to_datetime("2025-09-23 00:00:00")
    df_combined = df_combined[df_combined["timestamp"] >= test_start_date].reset_index(drop=True)
    return df_combined

def main():
    print("🚀 Bắt đầu đánh giá mô hình LightGBM & XGBoost đã huấn luyện...")
    
    # 1. Load và gộp dữ liệu
    df_raw = load_and_combine_datasets()
    
    # 2. Sinh đặc trưng nâng cao
    print("\n🛠️ Chạy Feature Engineering...")
    fe = FeatureEngineer()
    df_feat = fe.transform(df_raw)
    del df_raw
    gc.collect()
    
    # Ép kiểu float64 sang float32 để tiết kiệm RAM tối đa
    float64_cols = df_feat.select_dtypes(include=["float64"]).columns
    df_feat[float64_cols] = df_feat[float64_cols].astype("float32")
    gc.collect()
    
    lgbm_results = {}
    xgb_results = {}
    
    # Thư mục chứa model
    light_dir = "light"
    xg_dir = "xg"
    if not os.path.exists(light_dir):
        light_dir = "../light"
    if not os.path.exists(xg_dir):
        xg_dir = "../xg"
        
    # 3. Đánh giá từng chất khí
    for target in TARGET_COLUMNS:
        disp_name = TARGET_DISPLAY_NAMES[target]
        print(f"\n📊 Đánh giá chất khí: {disp_name}...")
        
        # Đặc trưng giống như lúc train
        raw_feature_cols = fe.get_feature_columns(df_feat)
        source_cols = ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]
        
        feature_cols = []
        for c in raw_feature_cols:
            if any(c.startswith(f"{g}_t+") for g in TARGET_COLUMNS):
                continue
            dtype = df_feat[c].dtype
            if dtype in [np.float32, np.float64, np.int32, np.int64, int, float] or dtype.name == "category":
                feature_cols.append(c)
                
        cols_to_keep = list(set(feature_cols + ["city", "station_id", "timestamp", target] + source_cols))
        cols_to_keep = [c for c in cols_to_keep if c in df_feat.columns]
        
        df_target = df_feat[cols_to_keep].copy()
        for col in source_cols:
            if col in df_target.columns:
                df_target[col] = df_target[col].astype("category")
                
        # Tạo dữ liệu trượt dự báo 24h tương lai (y) bằng shift(-h)
        y_cols = []
        for h in range(1, 25):
            col_name = f"{target}_t+{h}"
            df_target[col_name] = df_target.groupby(["city", "station_id"])[target].shift(-h)
            y_cols.append(col_name)
            
        df_clean = df_target.dropna(subset=feature_cols + y_cols).reset_index(drop=True)
        
        # Lọc tập Test (timestamp >= 2025-10-01)
        df_test = df_clean[df_clean["timestamp"] >= pd.to_datetime("2025-10-01 00:00:00")]
        
        # Tập Test Gộp (HN + HCM)
        X_test_all = df_test[feature_cols].copy()
        y_test_all = df_test[y_cols].copy().values
        
        # Đánh giá LightGBM
        lgbm_path = os.path.join(light_dir, f"lgbm_{target}.joblib")
        if os.path.exists(lgbm_path):
            lgbm_models = joblib.load(lgbm_path)
            
            # Dự báo Gộp
            y_pred_lgbm_all = np.zeros((len(X_test_all), 24))
            for idx, model_h in enumerate(lgbm_models):
                y_pred_lgbm_all[:, idx] = model_h.predict(X_test_all)
            y_pred_lgbm_all = np.clip(y_pred_lgbm_all, 0.0, None)
            
            mae_lgb = mean_absolute_error(y_test_all, y_pred_lgbm_all)
            rmse_lgb = np.sqrt(mean_squared_error(y_test_all, y_pred_lgbm_all))
            r2_lgb = r2_score(y_test_all.flatten(), y_pred_lgbm_all.flatten())
            
            lgbm_results[disp_name] = {
                "mae": mae_lgb,
                "rmse": rmse_lgb,
                "r2": r2_lgb
            }
        else:
            print(f"   ⚠️ Không tìm thấy model LightGBM tại {lgbm_path}")
            
        # Đánh giá XGBoost
        xgb_path = os.path.join(xg_dir, f"xgb_{target}.joblib")
        if os.path.exists(xgb_path):
            xgb_models = joblib.load(xgb_path)
            
            # Ép kiểu float64 sang float32 cho XGBoost
            X_test_all_xgb = X_test_all.copy()
            float_cols_all = X_test_all_xgb.select_dtypes(include=['float64']).columns
            X_test_all_xgb[float_cols_all] = X_test_all_xgb[float_cols_all].astype('float32')
            
            # Dự báo Gộp
            y_pred_xgb_all = np.zeros((len(X_test_all_xgb), 24))
            for idx, model_h in enumerate(xgb_models):
                y_pred_xgb_all[:, idx] = model_h.predict(X_test_all_xgb)
            y_pred_xgb_all = np.clip(y_pred_xgb_all, 0.0, None)
            
            mae_xg = mean_absolute_error(y_test_all, y_pred_xgb_all)
            rmse_xg = np.sqrt(mean_squared_error(y_test_all, y_pred_xgb_all))
            r2_xg = r2_score(y_test_all.flatten(), y_pred_xgb_all.flatten())
            
            xgb_results[disp_name] = {
                "mae": mae_xg,
                "rmse": rmse_xg,
                "r2": r2_xg
            }
        else:
            print(f"   ⚠️ Không tìm thấy model XGBoost tại {xgb_path}")
            
        del df_target, df_clean, df_test, X_test_all, y_test_all
        gc.collect()

    # 4. In bảng kết quả tổng hợp và lưu ra file
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("📊 BẢNG TỔNG HỢP HIỆU NĂNG TRÊN TẬP TEST 10% (HN + HCM)")
    output_lines.append("=" * 90)
    output_lines.append(f"{'Chất khí':<10} | {'LGBM MAE':<10} {'LGBM RMSE':<10} {'LGBM R²':<10} | {'XGB MAE':<10} {'XGB RMSE':<10} {'XGB R²':<10}")
    output_lines.append("-" * 90)
    for gas in TARGET_DISPLAY_NAMES.values():
        res_l = lgbm_results.get(gas, {})
        res_x = xgb_results.get(gas, {})
        
        l_mae = res_l.get("mae", np.nan)
        l_rmse = res_l.get("rmse", np.nan)
        l_r2 = res_l.get("r2", np.nan)
        
        x_mae = res_x.get("mae", np.nan)
        x_rmse = res_x.get("rmse", np.nan)
        x_r2 = res_x.get("r2", np.nan)
        output_lines.append(f"{gas:<10} | {l_mae:<10.2f} {l_rmse:<10.2f} {l_r2:<10.3f} | {x_mae:<10.2f} {x_rmse:<10.2f} {x_r2:<10.3f}")
    output_lines.append("=" * 90)
    
    output_text = "\n".join(output_lines)
    print("\n" + output_text)
    
    with open("trees_test_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("  → Đã lưu kết quả đánh giá vào file: trees_test_results.txt")

if __name__ == "__main__":
    main()
