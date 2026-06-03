"""
tune_lightgbm.py
================
Kịch bản tối ưu hóa siêu tham số (Hyperparameter Tuning) bằng Optuna cho LightGBM.
Áp dụng tỷ lệ chia dữ liệu 80 - 10 - 10:
- Train: 2020-01-01 đến 2024-12-31
- Val: 2025-01-01 đến 2025-09-30 (để Optuna đánh giá và chọn tham số)
- Test: 2025-10-01 đến 2026-04-06 (để đánh giá cuối cùng)
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tune_lightgbm")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    print("❌ LỖI: Chưa cài đặt thư viện 'lightgbm'. Vui lòng chạy: pip install lightgbm")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
except ImportError:
    print("❌ LỖI: Chưa cài đặt thư viện 'optuna'. Vui lòng chạy: pip install optuna")
    sys.exit(1)
import joblib
from sklearn.metrics import mean_absolute_error
from features import FeatureEngineer

def load_and_combine_datasets():
    """Đọc dữ liệu Hà Nội và TP.HCM rồi gộp lại làm một giống như TFT."""
    hn_path = "/kaggle/input/datasets/arisene/train-final/dataset_hanoi.parquet"
    hcm_path = "/kaggle/input/datasets/arisene/train-final/dataset_hcmc.parquet"
    
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "dataset_hanoi.parquet"
        hcm_path = "dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        raise FileNotFoundError(f"❌ Thiếu file dữ liệu tại {hn_path} hoặc {hcm_path}.")
        
    print(f"📖 Đang đọc dữ liệu Hà Nội từ: {hn_path}...")
    df_hn = pd.read_parquet(hn_path)
    df_hn["city"] = "hanoi"
    
    print(f"📖 Đang đọc dữ liệu TP.HCM từ: {hcm_path}...")
    df_hcm = pd.read_parquet(hcm_path)
    df_hcm["city"] = "hcmc"
    
    # Gộp 2 thành phố
    df_combined = pd.concat([df_hn, df_hcm], ignore_index=True)
    df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    
    print(f"✅ Gộp dữ liệu thành công! Tổng số dòng: {len(df_combined):,}")
    return df_combined

TARGET_COLUMNS = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
REPRESENTATIVE_HORIZONS = [1, 6, 12, 24]

def tune_gas(df_feat, target_gas, n_trials=30):
    logger.info(f"🚀 Bắt đầu tối ưu hóa siêu tham số cho: {target_gas} với {n_trials} trials")
    
    # 1. Lấy danh sách đặc trưng từ FeatureEngineer
    fe = FeatureEngineer()
    raw_feature_cols = fe.get_feature_columns(df_feat)
    
    # Ép kiểu source sang category
    source_cols = ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]
            
    # Lọc đặc trưng số và category
    feature_cols = []
    for c in raw_feature_cols:
        # Không lấy các cột target t+ của các khí khác
        if any(c.startswith(f"{g}_t+") for g in TARGET_COLUMNS):
            continue
        dtype = df_feat[c].dtype
        if dtype in [np.float32, np.float64, np.int32, np.int64, int, float] or dtype.name == "category":
            feature_cols.append(c)
            
    # Chỉ copy các cột thực sự cần dùng để giảm tải RAM tối đa
    cols_to_keep = list(set(feature_cols + ["city", "station_id", "timestamp", target_gas] + source_cols))
    cols_to_keep = [c for c in cols_to_keep if c in df_feat.columns]
    
    df_target = df_feat[cols_to_keep].copy()
    
    for col in source_cols:
        if col in df_target.columns:
            df_target[col] = df_target[col].astype("category")
            
    # 2. Tạo targets cho các bước đại diện
    y_cols = []
    for h in REPRESENTATIVE_HORIZONS:
        col_name = f"{target_gas}_t+{h}"
        df_target[col_name] = df_target.groupby(["city", "station_id"])[target_gas].shift(-h)
        y_cols.append(col_name)
        
    # Drop rows with NaN targets/features
    df_clean = df_target.dropna(subset=feature_cols + y_cols).reset_index(drop=True)
    
    # Chia dữ liệu theo mốc thời gian 80 - 10 - 10
    df_train = df_clean[df_clean["timestamp"] < pd.to_datetime("2025-01-01 00:00:00")]
    df_val = df_clean[(df_clean["timestamp"] >= pd.to_datetime("2025-01-01 00:00:00")) & 
                      (df_clean["timestamp"] < pd.to_datetime("2025-10-01 00:00:00"))]
                      
    # Lọc tập Val Hà Nội để tối ưu hóa trực tiếp trên Hà Nội
    df_val_hn = df_val[df_val["city"] == "hanoi"]
    
    X_train = df_train[feature_cols].copy()
    y_train = df_train[y_cols].copy()
    X_val_hn = df_val_hn[feature_cols].copy()
    y_val_hn = df_val_hn[y_cols].copy()
    
    # Giải phóng các dataframe trung gian ngay lập tức
    del df_target
    del df_clean
    del df_train
    del df_val
    del df_val_hn
    import gc
    gc.collect()
    
    # 4. Định nghĩa hàm mục tiêu cho Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        
        maes = []
        for h in REPRESENTATIVE_HORIZONS:
            col_name = f"{target_gas}_t+{h}"
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train[col_name])
            
            y_pred = model.predict(X_val_hn)
            y_pred = np.clip(y_pred, 0.0, None)
            
            mae = mean_absolute_error(y_val_hn[col_name], y_pred)
            maes.append(mae)
            
        gc.collect()
        return np.mean(maes)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"✨ Tối ưu hoàn tất cho {target_gas}!")
    logger.info(f"   Best MAE = {study.best_value:.4f}")
    logger.info(f"   Best params: {study.best_params}")
    
    # 5. Lưu bộ siêu tham số tốt nhất vào models/lightgbm/
    model_dir = "models/lightgbm"
    os.makedirs(model_dir, exist_ok=True)
    best_params_path = os.path.join(model_dir, f"best_params_{target_gas}.json")
    
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4, ensure_ascii=False)
        
    logger.info(f"💾 Đã lưu bộ siêu tham số tốt nhất vào: {best_params_path}")
    
    # Giải phóng dữ liệu huấn luyện của khí hiện tại
    del X_train
    del y_train
    del X_val_hn
    del y_val_hn
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tự động tìm kiếm siêu tham số bằng Optuna cho LightGBM.")
    parser.add_argument("--gas", type=str, default="pm25_obs", choices=TARGET_COLUMNS + ["all"],
                        help="Tên cột chất khí cần tối ưu hóa hoặc 'all' để tối ưu hóa toàn bộ 6 khí.")
    parser.add_argument("--trials", type=int, default=30,
                        help="Số lượng lượt chạy thử nghiệm của Optuna cho mỗi chất khí.")
                        
    args = parser.parse_args()
    
    # 1. Load và gộp dữ liệu thô (chỉ chạy 1 lần duy nhất)
    df_raw = load_and_combine_datasets()
    
    # 2. Sinh đặc trưng nâng cao bằng FeatureEngineer
    logger.info("🛠️ Đang chạy Feature Engineering cho toàn bộ dữ liệu...")
    fe = FeatureEngineer()
    df_feat = fe.transform(df_raw)
    
    # Giải phóng raw data
    del df_raw
    import gc
    gc.collect()
    
    if args.gas == "all":
        logger.info("🌟 BẮT ĐẦU TỐI ƯU HÓA SIÊU THAM SỐ CHO TOÀN BỘ 6 CHẤT KHÍ 🌟")
        for gas in TARGET_COLUMNS:
            tune_gas(df_feat, gas, args.trials)
        logger.info("🎉 ĐÃ HOÀN TẤT TỐI ƯU HÓA CHO TOÀN BỘ 6 CHẤT KHÍ!")
    else:
        tune_gas(df_feat, args.gas, args.trials)
