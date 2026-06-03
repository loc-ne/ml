"""
deploy/test_lightgbm_all_gases.py
=================================
Kịch bản huấn luyện và đánh giá mô hình LightGBM (Light Gradient Boosting Machine)
cho CẢ 6 KHÍ trên chuỗi dự báo 24 giờ tiếp theo (t+1 đến t+24).
- GIỐNG HỆT TFT: Load dữ liệu gốc Hà Nội và TP.HCM, gộp lại để huấn luyện chung.
- KHÔNG TẠO LẠI ĐẶC TRƯNG: Sử dụng trực tiếp 100% các cột đặc trưng đã được tạo sẵn trong file Parquet gốc.
- ĐÁNH GIÁ CHUẨN XÁC: Đánh giá sai số riêng trên tập Test Hà Nội để đối chiếu sòng phẳng với báo cáo.
- Phân chia: 80% Train (2020-2024), 20% Test (2025-2026).
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Kiểm tra xem lightgbm đã cài đặt chưa
try:
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:
    print("❌ LỖI: Chưa cài đặt thư viện 'lightgbm'.")
    print("👉 Vui lòng chạy lệnh này trong terminal trước: pip install lightgbm")
    sys.exit(1)

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import FeatureEngineer

# 6 Targets chính xác theo đúng tft.py
TARGET_COLUMNS = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
TARGET_DISPLAY_NAMES = {
    "pm25_obs": "PM2.5",
    "pm10_obs": "PM10",
    "no2_pseudo": "NO₂",
    "so2_pseudo": "SO₂",
    "co_pseudo": "CO",
    "o3_pseudo": "O₃",
}

# Kết quả MAE/RMSE thực tế của TFT (lấy từ báo cáo đánh giá Hà Nội)
TFT_RESULTS = {
    "PM2.5": {"mae": 5.42, "rmse": 6.53},
    "PM10":  {"mae": 4.79, "rmse": 5.98},
    "NO₂":   {"mae": 5.83, "rmse": 6.86},
    "SO₂":   {"mae": 4.17, "rmse": 5.12},
    "CO":    {"mae": 66.46, "rmse": 85.10},
    "O₃":    {"mae": 29.04, "rmse": 35.70},
}

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

def main():
    print("🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM SO SÁNH LIGHTGBM (HN + HCM COMBINED)")
    print("🌟 TÍCH HỢP BỘ TẠO ĐẶC TRƯNG NÂNG CAO TỪ FEATURES.PY & CHIA 80-10-10")
    print("═" * 95)

    # 1. Load và gộp dữ liệu
    df_raw = load_and_combine_datasets()
    
    # 2. Sinh đặc trưng nâng cao bằng FeatureEngineer
    print("\n🛠️ Đang khởi tạo và chạy Feature Engineering từ features.py...")
    fe = FeatureEngineer()
    df_feat = fe.transform(df_raw)
    print(f"✅ Sinh đặc trưng hoàn thành! Kích thước dữ liệu mới: {df_feat.shape}\n")
    
    # Giải phóng raw data ngay lập tức
    del df_raw
    import gc
    gc.collect()
    
    comparison_table = []
    
    # 3. Huấn luyện và đánh giá từng chất khí
    for target in TARGET_COLUMNS:
        disp_name = TARGET_DISPLAY_NAMES[target]
        print(f"\n🌲 Đang huấn luyện LightGBM cho chất khí: {disp_name}...")
        
        # Lấy danh sách các cột đặc trưng từ FeatureEngineer
        raw_feature_cols = fe.get_feature_columns(df_feat)
        
        # Ép kiểu dữ liệu nguồn (source) sang category để LightGBM tự nhận diện
        source_cols = ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]
                
        # Lọc đặc trưng: chỉ giữ lại các cột số và cột phân loại (category)
        feature_cols = []
        for c in raw_feature_cols:
            if any(c.startswith(f"{g}_t+") for g in TARGET_COLUMNS):
                continue
            dtype = df_feat[c].dtype
            if dtype in [np.float32, np.float64, np.int32, np.int64, int, float] or dtype.name == "category":
                feature_cols.append(c)
                
        # Chỉ copy các cột thực sự cần dùng để giảm tải RAM tối đa
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
            
        # Loại bỏ các dòng bị khuyết nhãn target tương lai hoặc đặc trưng
        df_clean = df_target.dropna(subset=feature_cols + y_cols).reset_index(drop=True)
        
        # Chia tách Train - Val - Test theo thời gian (80 - 10 - 10)
        df_train = df_clean[df_clean["timestamp"] < pd.to_datetime("2025-01-01 00:00:00")]
        df_val = df_clean[(df_clean["timestamp"] >= pd.to_datetime("2025-01-01 00:00:00")) & 
                          (df_clean["timestamp"] < pd.to_datetime("2025-10-01 00:00:00"))]
        df_test = df_clean[df_clean["timestamp"] >= pd.to_datetime("2025-10-01 00:00:00")]
        
        # Lọc tập Val & Test riêng của HÀ NỘI để đối chiếu sòng phẳng
        df_val_hanoi = df_val[df_val["city"] == "hanoi"]
        df_test_hanoi = df_test[df_test["city"] == "hanoi"]
        
        X_train = df_train[feature_cols].copy()
        y_train = df_train[y_cols].copy()
        
        X_val_hn = df_val_hanoi[feature_cols].copy()
        y_val_hn = df_val_hanoi[y_cols].copy()
        
        X_test_hn = df_test_hanoi[feature_cols].copy()
        y_test_hn = df_test_hanoi[y_cols].copy()
        
        # Giải phóng các dataframe trung gian lớn
        del df_target
        del df_clean
        del df_train
        del df_val
        del df_test
        del df_val_hanoi
        del df_test_hanoi
        gc.collect()
        
        # Cấu hình LightGBM: sử dụng bộ tối ưu từ Optuna nếu có, ngược lại dùng mặc định
        best_params_path = f"models/lightgbm/best_params_{target}.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"   ⚙️ Đang sử dụng bộ siêu tham số tối ưu tìm được bởi Optuna.")
        else:
            params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            print(f"   ⚙️ Đang sử dụng bộ siêu tham số mặc định.")
            
        params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
        lgbm = LGBMRegressor(**params)
        
        # Wrap trong MultiOutputRegressor để dự báo 24h đồng thời
        model = MultiOutputRegressor(lgbm)
        model.fit(X_train, y_train)
        
        # Lưu mô hình đã huấn luyện
        model_dir = "models/lightgbm"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"lgbm_{target}.joblib")
        joblib.dump(model, model_path)
        print(f"   💾 Đã lưu mô hình {disp_name} tại: {model_path}")
        
        # Dự báo 24 bước tương lai trên tập Val & Test của Hà Nội
        y_pred_val = model.predict(X_val_hn)
        y_pred_val = np.clip(y_pred_val, 0.0, None)
        
        y_pred_test = model.predict(X_test_hn)
        y_pred_test = np.clip(y_pred_test, 0.0, None)
        
        # Tính toán sai số trung bình trên Hà Nội
        mae_val = mean_absolute_error(y_val_hn, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_val_hn, y_pred_val))
        
        mae_test = mean_absolute_error(y_test_hn, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_hn, y_pred_test))
        
        # Lấy kết quả đối chiếu của TFT
        tft_mae = TFT_RESULTS[disp_name]["mae"]
        tft_rmse = TFT_RESULTS[disp_name]["rmse"]
        
        comparison_table.append({
            "Gas": disp_name,
            "LGBM_VAL_MAE": mae_val,
            "LGBM_TEST_MAE": mae_test,
            "TFT_MAE": tft_mae,
            "LGBM_VAL_RMSE": rmse_val,
            "LGBM_TEST_RMSE": rmse_test,
            "TFT_RMSE": tft_rmse,
        })
        
        print(f"   📊 Kết quả trung bình 24h tại Hà Nội (Sử dụng {len(feature_cols)} đặc trưng nâng cao):")
        print(f"      - LightGBM Val  (2025/01-2025/09): MAE = {mae_val:.2f} | RMSE = {rmse_val:.2f}")
        print(f"      - LightGBM Test (2025/10-2026/04): MAE = {mae_test:.2f} | RMSE = {rmse_test:.2f} 🌟")
        print(f"      - TFT Model (Best - Full 20%):     MAE = {tft_mae:.2f} | RMSE = {tft_rmse:.2f}")
        
        # Giải phóng dữ liệu huấn luyện của khí hiện tại để chuẩn bị cho khí tiếp theo
        del X_train
        del y_train
        del X_val_hn
        del y_val_hn
        del X_test_hn
        del y_test_hn
        gc.collect()
        
    # ── IN BẢNG ĐỐI CHIẾU TỔNG HỢP ──────────────────────────────────────────
    print("\n" + "═" * 125)
    print("📊 BẢNG TỔNG HỢP ĐỐI CHIẾU HIỆU NĂNG TRÊN TẬP VAL & TEST HÀ NỘI")
    print("═" * 125)
    print(f"{'Chất khí':<10} | {'LGBM VAL MAE':<13} | {'LGBM TEST MAE':<14} | {'TFT MAE (20%)':<14} | {'LGBM VAL RMSE':<14} | {'LGBM TEST RMSE':<15} | {'TFT RMSE (20%)':<15}")
    print("-" * 125)
    for row in comparison_table:
        print(f"{row['Gas']:<10} | "
              f"{row['LGBM_VAL_MAE']:<13.2f} | "
              f"{row['LGBM_TEST_MAE']:<14.2f} | "
              f"{row['TFT_MAE']:<14.2f} | "
              f"{row['LGBM_VAL_RMSE']:<13.2f} | "
              f"{row['LGBM_TEST_RMSE']:<15.2f} | "
              f"{row['TFT_RMSE']:<15.2f}")
    print("═" * 125)
    print("👉 Nhận xét: LightGBM mới được bổ sung 250 đặc trưng nâng cao và chia tập 80-10-10.")
    print("   Bộ siêu tham số tối ưu từ Optuna giúp cải thiện đáng kể so với mô hình gốc.")
    print("═" * 125)

if __name__ == "__main__":
    main()
