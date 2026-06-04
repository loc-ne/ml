"""
evaluate_tft.py
===============
Kịch bản đánh giá độc lập mô hình Temporal Fusion Transformer (TFT) trên tập Test.
- Tải mô hình từ file checkpoint (.ckpt) đã lưu.
- Khôi phục cấu trúc Dataset bằng file training_dataset_params.pkl.
- Tính toán chi tiết sai số MAE và RMSE trên tập Test (từ 2025-10-01 trở đi).
- Đánh giá trên cả tập Test gộp và tập Test riêng cho Hà Nội để so sánh trực tiếp với Baseline.
"""

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sys
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

def evaluate_model(checkpoint_path="models/tft-best-model.ckpt"):
    print("🚀 Bắt đầu tiến trình đánh giá mô hình TFT trên tập Test...")
    
    # 1. Đọc dữ liệu gốc
    hn_path = "/kaggle/input/datasets/arisene/train-final/dataset_hanoi.parquet"
    hcm_path = "/kaggle/input/datasets/arisene/train-final/dataset_hcmc.parquet"
    
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "dataset_hanoi.parquet"
        hcm_path = "dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu {hn_path} hoặc {hcm_path}.")
        sys.exit(1)
        
    df_hn = pd.read_parquet(hn_path)
    df_hn["city"] = "hanoi"
    df_hcm = pd.read_parquet(hcm_path)
    df_hcm["city"] = "hcmc"
    
    df = pd.concat([df_hn, df_hcm], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    df["time_idx"] = df["time_idx"].astype(int)
    
    # Ép kiểu các cột phân loại
    categorical_cols = [
        "city", "station_id", "is_weekend", "is_holiday", "is_tet", 
        "is_peak_traffic", "is_winter_north", "is_harvest_north",
        "is_hot_dry_south", "is_dry_season_south", "is_harvest_season",
        "is_foggy_risk", "is_smog_risk", "is_day", "data_source",
        "pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    # Xử lý trễ dữ liệu vệ tinh S5P (khớp với tft.py)
    satellite_features = [
        "s5p_no2", "s5p_no2_cf", "s5p_so2", "s5p_so2_cf", "s5p_co", "s5p_co_cf", 
        "s5p_o3", "s5p_o3_cf", "s5p_aai", "s5p_aai_cf", "s5p_aod", "s5p_aod_cf", 
        "s5p_days_since_obs", "s5p_wind_alignment"
    ]
    print("⏳ Đang xử lý độ trễ thực tế 24h của dữ liệu vệ tinh S5P...")
    cols_to_shift = [col for col in satellite_features if col in df.columns]
    if cols_to_shift:
        grouped = df.groupby(["city", "station_id"])
        df[cols_to_shift] = grouped[cols_to_shift].shift(24)
        df[cols_to_shift] = grouped[cols_to_shift].ffill()
        df[cols_to_shift] = grouped[cols_to_shift].bfill()

    # 2. Khôi phục dataset parameters
    params_path = "models/training_dataset_params.pkl"
    if not os.path.exists(params_path):
        print(f"❌ LỖI: Không tìm thấy file lưu cấu trúc Dataset tại {params_path}.")
        print("Vui lòng đảm bảo bạn đã lưu file này từ tiến trình train.")
        sys.exit(1)
        
    with open(params_path, "rb") as f:
        dataset_params = pickle.load(f)
        
    # Xác định mốc cutoff của tập Test (2025-10-01)
    t_min = df["timestamp"].min()
    validation_cutoff = int((pd.to_datetime("2025-10-01 00:00:00") - t_min).total_seconds() // 3600)
    
    # 3. Tạo Test Dataset cho cả 2 thành phố
    test_dataset = TimeSeriesDataSet.from_parameters(
        dataset_params, 
        df, 
        min_prediction_idx=validation_cutoff + 1, 
        stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=1024, num_workers=2)
    
    # 4. Tạo Test Dataset chỉ cho Hà Nội (để đối chiếu sòng phẳng với Baseline)
    df_hn_only = df[df.city == "hanoi"].reset_index(drop=True)
    test_dataset_hn = TimeSeriesDataSet.from_parameters(
        dataset_params,
        df_hn_only,
        min_prediction_idx=validation_cutoff + 1,
        stop_randomization=True
    )
    test_dataloader_hn = test_dataset_hn.to_dataloader(train=False, batch_size=1024, num_workers=2)
    
    # 5. Tải mô hình tốt nhất
    if not os.path.exists(checkpoint_path):
        # Thử tìm file .ckpt tự động trong thư mục models/
        print(f"⚠️ Không thấy checkpoint tại {checkpoint_path}. Đang tìm kiếm trong thư mục models/...")
        ckpt_files = [f for f in os.listdir("models") if f.endswith(".ckpt")]
        if ckpt_files:
            checkpoint_path = os.path.join("models", sorted(ckpt_files)[-1]) # Lấy file mới nhất
            print(f"📂 Tìm thấy file checkpoint thay thế: {checkpoint_path}")
        else:
            print("❌ LỖI: Không tìm thấy bất kỳ file checkpoint .ckpt nào trong thư mục models/.")
            sys.exit(1)
            
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📖 Đang tải mô hình từ checkpoint: {checkpoint_path} trên thiết bị {device.upper()}...")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    
    # 6. Tính toán sai số
    target_columns = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
    
    def calculate_metrics(dataloader, description="TẬP TEST GỘP (HN + HCM)"):
        print(f"📊 Đang tính toán sai số cho: {description}...")
        with torch.no_grad():
            predictions = model.predict(dataloader, return_y=True)
            
        print(f"{'Chất khí':<15} | {'MAE':<10} | {'RMSE':<10}")
        print("─" * 40)
        
        for i, target in enumerate(target_columns):
            disp_name = target.replace("_obs", "").replace("_pseudo", "").upper()
            
            # Lấy predictions và actuals
            if isinstance(predictions, tuple) and (isinstance(predictions[0], list) or isinstance(predictions[0], tuple)):
                pred_vals = predictions[0][i].float()
                actual_vals = predictions[1][i].float()
            elif isinstance(predictions, tuple):
                pred_vals = predictions[0].float()
                actual_vals = predictions[1].float()
            else:
                pred_vals = predictions.float()
                actual_vals = None
                
            if actual_vals is not None:
                mae = torch.mean(torch.abs(pred_vals - actual_vals)).item()
                rmse = torch.sqrt(torch.mean((pred_vals - actual_vals) ** 2)).item()
                print(f"{disp_name:<15} | {mae:<10.4f} | {rmse:<10.4f}")
                
    # Chạy tính toán
    calculate_metrics(test_dataloader, "TẬP TEST GỘP (HN + HCM)")
    calculate_metrics(test_dataloader_hn, "TẬP TEST RIÊNG HÀ NỘI (ĐỐI CHIẾU BASELINE)")

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else "models/tft-best-model.ckpt"
    evaluate_model(path_arg)
