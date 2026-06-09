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
import gc

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
    hn_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hanoi.parquet"
    hcm_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hcmc.parquet"
    
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
    
    # Ép kiểu float64 sang float32 để giảm tải RAM tối đa
    float64_cols = df_feat.select_dtypes(include=["float64"]).columns
    df_feat[float64_cols] = df_feat[float64_cols].astype("float32")
    
    # Giải phóng raw data ngay lập tức
    del df_raw
    gc.collect()
    
    # Ép kiểu danh mục (category) cho các cột phân loại trong df_feat
    source_cols = ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]
    for col in source_cols + ["station_id_encoded", "city"]:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("category")
    gc.collect()
    
    comparison_table = []
    all_losses = {}
    
    # 3. Huấn luyện và đánh giá từng chất khí
    for target in TARGET_COLUMNS:
        disp_name = TARGET_DISPLAY_NAMES[target]
        print(f"\n🌲 Đang huấn luyện LightGBM cho chất khí: {disp_name}...")
        
        # Lấy danh sách các cột đặc trưng từ FeatureEngineer
        raw_feature_cols = fe.get_feature_columns(df_feat)
                
        # Lọc đặc trưng: chỉ giữ lại các cột số và cột phân loại (category)
        feature_cols = []
        for c in raw_feature_cols:
            if any(c.startswith(f"{g}_t+") for g in TARGET_COLUMNS):
                continue
            dtype = df_feat[c].dtype
            if dtype in [np.float32, np.float64, np.int32, np.int64, int, float] or dtype.name == "category":
                feature_cols.append(c)
                
        # Bổ sung city vào features
        if "city" in df_feat.columns:
            feature_cols.append("city")
            
        # Tạo dữ liệu trượt dự báo 24h tương lai (y) bằng shift(-h) trên dataframe nhỏ để tiết kiệm RAM
        df_y = df_feat[["city", "station_id", target, "timestamp"]].copy()
        y_cols = []
        for h in range(1, 25):
            col_name = f"{target}_t+{h}"
            df_y[col_name] = df_y.groupby(["city", "station_id"])[target].shift(-h)
            y_cols.append(col_name)
            
        # Loại bỏ các dòng bị khuyết nhãn target tương lai
        df_y = df_y.dropna(subset=y_cols)
        valid_idx = df_y.index
        
        # Chia tách Train - Val - Test theo thời gian (80 - 10 - 10) dựa trên chỉ mục
        train_mask = df_y["timestamp"] < pd.to_datetime("2025-01-01 00:00:00")
        val_mask = (df_y["timestamp"] >= pd.to_datetime("2025-01-01 00:00:00")) & (df_y["timestamp"] < pd.to_datetime("2025-10-01 00:00:00"))
        test_mask = df_y["timestamp"] >= pd.to_datetime("2025-10-01 00:00:00")
        
        # Lọc tập Val Hà Nội (để theo dõi/đối chiếu) và tập Test Gộp (HN + HCM)
        val_hn_mask = val_mask & (df_y["city"] == "hanoi")
        
        X_train = df_feat.loc[valid_idx[train_mask], feature_cols].copy()
        y_train = df_y.loc[valid_idx[train_mask], y_cols].copy()
        
        X_val_hn = df_feat.loc[valid_idx[val_hn_mask], feature_cols].copy()
        y_val_hn = df_y.loc[valid_idx[val_hn_mask], y_cols].copy()
        
        X_test_all = df_feat.loc[valid_idx[test_mask], feature_cols].copy()
        y_test_all = df_y.loc[valid_idx[test_mask], y_cols].copy()
        
        # Giải phóng dataframe phụ df_y ngay lập tức
        del df_y
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
        
        # Train 24 mô hình độc lập cho 24 bước thời gian để vẽ Learning Curve
        models = []
        train_losses_all = []
        val_losses_all = []
        
        print(f"   ⚙️ Đang huấn luyện và theo dõi quá trình học (Learning Curve) cho 24 horizons...")
        for h in range(1, 25):
            print(f"      → [Horizon {h:02d}/24] Đang huấn luyện {disp_name} t+{h}...", flush=True)
            col_name = f"{target}_t+{h}"
            model_h = LGBMRegressor(**params)
            
            # Huấn luyện và thu thập log học tập
            model_h.fit(
                X_train, y_train[col_name],
                eval_set=[(X_train, y_train[col_name]), (X_val_hn, y_val_hn[col_name])],
                eval_metric="mae",
                callbacks=[]
            )
            
            evals_result = model_h.evals_result_
            train_losses_all.append(evals_result['training']['l1'])
            val_losses_all.append(evals_result['valid_1']['l1'])
            
            models.append(model_h)
            
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
        plt.title(f"Learning Curve for {disp_name} (Average over 24 horizons)", fontsize=12, fontweight="bold")
        plt.xlabel("Boosting Rounds", fontsize=10)
        plt.ylabel("MAE", fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        model_dir = "models/lightgbm"
        os.makedirs(model_dir, exist_ok=True)
        plot_path = os.path.join(model_dir, f"learning_curve_{target}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   📈 Đã vẽ và lưu biểu đồ quá trình học tại: {plot_path}")
        
        # Lưu toàn bộ danh sách 24 mô hình của chất khí này
        model_path = os.path.join(model_dir, f"lgbm_{target}.joblib")
        joblib.dump(models, model_path)
        print(f"   💾 Đã lưu mô hình {disp_name} tại: {model_path}")
        
        # Dự báo 24 bước tương lai trên tập Val Hà Nội và tập Test Gộp (HN + HCM)
        y_pred_val = np.zeros((len(X_val_hn), 24))
        y_pred_test_all = np.zeros((len(X_test_all), 24))
        
        for idx, model_h in enumerate(models):
            y_pred_val[:, idx] = model_h.predict(X_val_hn)
            y_pred_test_all[:, idx] = model_h.predict(X_test_all)
            
        y_pred_val = np.clip(y_pred_val, 0.0, None)
        y_pred_test_all = np.clip(y_pred_test_all, 0.0, None)
        
        # Tính toán sai số trung bình
        mae_val = mean_absolute_error(y_val_hn, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_val_hn, y_pred_val))
        
        mae_test = mean_absolute_error(y_test_all, y_pred_test_all)
        rmse_test = np.sqrt(mean_squared_error(y_test_all, y_pred_test_all))
        from sklearn.metrics import r2_score
        r2_test = r2_score(y_test_all.values.flatten(), y_pred_test_all.flatten())
        
        # Lấy kết quả đối chiếu của TFT
        tft_mae = TFT_RESULTS[disp_name]["mae"]
        tft_rmse = TFT_RESULTS[disp_name]["rmse"]
        
        comparison_table.append({
            "Gas": disp_name,
            "LGBM_VAL_MAE": mae_val,
            "LGBM_TEST_MAE": mae_test,
            "LGBM_TEST_RMSE": rmse_test,
            "LGBM_TEST_R2": r2_test,
            "TFT_MAE": tft_mae,
            "TFT_RMSE": tft_rmse,
        })
        
        print(f"   📊 Kết quả trung bình 24h trên TẬP TEST GỘP HN+HCM (Sử dụng {len(feature_cols)} đặc trưng nâng cao):")
        print(f"      - LightGBM Val Hà Nội: MAE = {mae_val:.2f} | RMSE = {rmse_val:.2f}")
        print(f"      - LightGBM Test Gộp:   MAE = {mae_test:.2f} | RMSE = {rmse_test:.2f} | R² = {r2_test:.3f} 🌟")
        print(f"      - TFT Model (Best):    MAE = {tft_mae:.2f} | RMSE = {tft_rmse:.2f}")
        
        # Giải phóng dữ liệu huấn luyện của khí hiện tại để chuẩn bị cho khí tiếp theo
        del X_train
        del y_train
        del X_val_hn
        del y_val_hn
        del X_test_all
        del y_test_all
        gc.collect()
        
    # ── IN BẢNG ĐỐI CHIẾU TỔNG HỢP VÀ GHI FILE ──────────────────────────────────────────
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("📊 BẢNG TỔNG HỢP HIỆU NĂNG LIGHTGBM TRÊN TẬP TEST 10% (HN + HCM)")
    output_lines.append("=" * 90)
    output_lines.append(f"{'Chất khí':<10} | {'LGBM MAE':<12} {'LGBM RMSE':<12} {'LGBM R²':<12} | {'TFT MAE':<12} {'TFT RMSE':<12}")
    output_lines.append("-" * 90)
    for row in comparison_table:
        output_lines.append(f"{row['Gas']:<10} | {row['LGBM_TEST_MAE']:<12.2f} {row['LGBM_TEST_RMSE']:<12.2f} {row['LGBM_TEST_R2']:<12.3f} | {row['TFT_MAE']:<12.2f} {row['TFT_RMSE']:<12.2f}")
    output_lines.append("=" * 90)
    
    output_text = "\n".join(output_lines)
    print("\n" + output_text)
    
    with open("lgbm_test_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("  → Đã lưu kết quả đánh giá vào file: lgbm_test_results.txt")

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
        combined_plot_path = os.path.join("models/lightgbm", "learning_curves_combined.png")
        plt.savefig(combined_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Đã vẽ và lưu biểu đồ gộp 6 chất khí thành công tại: {combined_plot_path}")

if __name__ == "__main__":
    main()
