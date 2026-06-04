"""
train_xgboost.py
================
Kịch bản huấn luyện và đánh giá mô hình XGBoost cho CẢ 6 KHÍ
trên chuỗi dự báo 24 giờ tiếp theo (t+1 đến t+24).
- GIỐNG HỆT TFT: Load dữ liệu gốc Hà Nội và TP.HCM, gộp lại để huấn luyện chung.
- ĐÁNH GIÁ CHUẨN XÁC: Đánh giá sai số riêng trên tập Test Hà Nội để đối chiếu sòng phẳng.
- Phân chia: 80% Train (< 2025-01-01), 10% Val (2025-01-01 đến < 2025-10-01), 10% Test (>= 2025-10-01).
- Tích hợp vẽ biểu đồ Learning Curve (MAE trung bình qua 24 horizons).
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

try:
    from xgboost import XGBRegressor
except ImportError:
    print("❌ LỖI: Chưa cài đặt thư viện 'xgboost'. Vui lòng chạy: pip install xgboost")
    sys.exit(1)

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import FeatureEngineer

# Tự động phát hiện GPU
try:
    import torch
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE_NAME = "cpu"

# 6 Targets chính xác theo tft.py
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH XGBOOST (HN + HCM COMBINED)")
    print(f"🖥️ Thiết bị sử dụng: {DEVICE_NAME.upper()}")
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
    all_losses = {}
    
    # 3. Huấn luyện và đánh giá từng chất khí
    for target in TARGET_COLUMNS:
        disp_name = TARGET_DISPLAY_NAMES[target]
        print(f"\n🌲 Đang huấn luyện XGBoost cho chất khí: {disp_name}...")
        
        # Lấy danh sách các cột đặc trưng từ FeatureEngineer
        raw_feature_cols = fe.get_feature_columns(df_feat)
        
        # Ép kiểu dữ liệu nguồn (source) sang category
        source_cols = ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]
                
        # Lọc đặc trưng số và category (khớp hoàn toàn với train_lightgbm.py)
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
        
        # Ép kiểu float64 sang float32 để giảm tải RAM 50% cho các ma trận đặc trưng
        for df_tmp in [X_train, X_val_hn, X_test_hn]:
            float_cols = df_tmp.select_dtypes(include=['float64']).columns
            df_tmp[float_cols] = df_tmp[float_cols].astype('float32')
            
        # Giải phóng các dataframe trung gian lớn
        del df_target
        del df_clean
        del df_train
        del df_val
        del df_test
        del df_val_hanoi
        del df_test_hanoi
        gc.collect()
        
        # Cấu hình XGBoost: sử dụng bộ tối ưu từ Optuna nếu có, ngược lại dùng mặc định
        best_params_path = f"models/xgboost/best_params_{target}.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"   ⚙️ Đang sử dụng bộ siêu tham số tối ưu tìm được bởi Optuna.")
        else:
            params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            print(f"   ⚙️ Đang sử dụng bộ siêu tham số mặc định.")
            
        params.update({
            "random_state": 42, 
            "n_jobs": -1,
            "tree_method": "hist",
            "device": DEVICE_NAME,
            "enable_categorical": True,
            "eval_metric": "mae"
        })
        
        # Train 24 mô hình độc lập cho 24 bước thời gian để vẽ Learning Curve
        models = []
        train_losses_all = []
        val_losses_all = []
        
        print(f"   ⚙️ Đang huấn luyện và theo dõi quá trình học (Learning Curve) cho 24 horizons...")
        for h in range(1, 25):
            col_name = f"{target}_t+{h}"
            model_h = XGBRegressor(**params)
            
            # Huấn luyện và thu thập log học tập trên toàn bộ tập dữ liệu (khớp với LightGBM)
            model_h.fit(
                X_train, y_train[col_name],
                eval_set=[(X_train, y_train[col_name]), (X_val_hn, y_val_hn[col_name])],
                verbose=False
            )
            
            evals_result = model_h.evals_result()
            train_losses_all.append(evals_result['validation_0']['mae'])
            val_losses_all.append(evals_result['validation_1']['mae'])
            
            models.append(model_h)
            gc.collect()
            
        # Vẽ và lưu biểu đồ Learning Curve (MAE trung bình qua 24 horizons)
        import matplotlib.pyplot as plt
        avg_train_loss = np.mean(train_losses_all, axis=0)
        avg_val_loss = np.mean(val_losses_all, axis=0)
        
        # Lưu kết quả để vẽ biểu đồ gộp ở cuối chương trình
        all_losses[target] = {
            "display_name": disp_name,
            "train": avg_train_loss,
            "val": avg_val_loss
        }
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(avg_train_loss) + 1), avg_train_loss, label="Train MAE", color="#3b82f6", linewidth=2)
        plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label="Val MAE", color="#ef4444", linewidth=2)
        plt.title(f"XGBoost Learning Curve for {disp_name} (Average over 24 horizons)", fontsize=12, fontweight="bold")
        plt.xlabel("Boosting Rounds", fontsize=10)
        plt.ylabel("MAE", fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        model_dir = "models/xgboost"
        os.makedirs(model_dir, exist_ok=True)
        plot_path = os.path.join(model_dir, f"learning_curve_{target}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   📈 Đã vẽ và lưu biểu đồ quá trình học tại: {plot_path}")
        
        # Lưu toàn bộ danh sách 24 mô hình của chất khí này
        model_path = os.path.join(model_dir, f"xgb_{target}.joblib")
        joblib.dump(models, model_path)
        print(f"   💾 Đã lưu mô hình {disp_name} tại: {model_path}")
        
        # Dự báo 24 bước tương lai trên tập Val & Test của Hà Nội
        y_pred_val = np.zeros((len(X_val_hn), 24))
        y_pred_test = np.zeros((len(X_test_hn), 24))
        
        for idx, model_h in enumerate(models):
            y_pred_val[:, idx] = model_h.predict(X_val_hn)
            y_pred_test[:, idx] = model_h.predict(X_test_hn)
            
        y_pred_val = np.clip(y_pred_val, 0.0, None)
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
            "XGB_VAL_MAE": mae_val,
            "XGB_TEST_MAE": mae_test,
            "TFT_MAE": tft_mae,
            "XGB_VAL_RMSE": rmse_val,
            "XGB_TEST_RMSE": rmse_test,
            "TFT_RMSE": tft_rmse,
        })
        
        print(f"   📊 Kết quả trung bình 24h tại Hà Nội (Sử dụng {len(feature_cols)} đặc trưng nâng cao):")
        print(f"      - XGBoost Val   (2025/01-2025/09): MAE = {mae_val:.2f} | RMSE = {rmse_val:.2f}")
        print(f"      - XGBoost Test  (2025/10-2026/04): MAE = {mae_test:.2f} | RMSE = {rmse_test:.2f} 🌟")
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
    print("📊 BẢNG TỔNG HỢP ĐỐI CHIẾU HIỆU NĂNG TRÊN TẬP VAL & TEST HÀ NỘI (XGBOOST VS TFT)")
    print("═" * 125)
    print(f"{'Chất khí':<10} | {'XGB VAL MAE':<13} | {'XGB TEST MAE':<14} | {'TFT MAE (20%)':<14} | {'XGB VAL RMSE':<14} | {'XGB TEST RMSE':<15} | {'TFT RMSE (20%)':<15}")
    print("-" * 125)
    for row in comparison_table:
        print(f"{row['Gas']:<10} | "
              f"{row['XGB_VAL_MAE']:<13.2f} | "
              f"{row['XGB_TEST_MAE']:<14.2f} | "
              f"{row['TFT_MAE']:<14.2f} | "
              f"{row['XGB_VAL_RMSE']:<13.2f} | "
              f"{row['XGB_TEST_RMSE']:<15.2f} | "
              f"{row['TFT_RMSE']:<15.2f}")
    print("═" * 125)
    print("👉 Nhận xét: XGBoost được huấn luyện với các đặc trưng nâng cao và tối ưu hóa siêu tham số từ Optuna.")
    print("═" * 125)

    # 4. Vẽ và lưu biểu đồ gộp 6 chất khí (2 hàng, 3 cột)
    if all_losses:
        print("\n📊 Đang vẽ và lưu biểu đồ gộp 6 chất khí (Combined Learning Curves)...")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, target in enumerate(TARGET_COLUMNS):
            if target in all_losses:
                data = all_losses[target]
                ax = axes[idx]
                ax.plot(range(1, len(data["train"]) + 1), data["train"], label="Train MAE", color="#3b82f6", linewidth=2)
                ax.plot(range(1, len(data["val"]) + 1), data["val"], label="Val MAE", color="#ef4444", linewidth=2)
                ax.set_title(f"Learning Curve for {data['display_name']}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Boosting Rounds", fontsize=9)
                ax.set_ylabel("MAE", fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.5)
                
        plt.tight_layout()
        combined_plot_path = os.path.join("models/xgboost", "learning_curves_combined.png")
        plt.savefig(combined_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Đã vẽ và lưu biểu đồ gộp 6 chất khí thành công tại: {combined_plot_path}")

if __name__ == "__main__":
    main()
