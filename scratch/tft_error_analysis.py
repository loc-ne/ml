"""
scratch/tft_error_analysis.py
==============================
Kịch bản độc lập chạy phân tích lỗi cho mô hình Temporal Fusion Transformer (TFT).
- Đọc tập Test và chạy dự báo t+12 cho cả 6 khí.
- Vẽ biểu đồ tán xạ gộp 2x3 actual vs predicted lưu tại result/tft_actual_vs_predicted.png.
- Trích xuất các ví dụ cụ thể dự đoán sai lệch nhiều nhất cho cả 6 khí (gồm Underprediction và Overprediction)
  và viết báo cáo markdown chi tiết kèm giả thuyết tại: result/tft_error_analysis.md
"""

import os
import sys
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# Fix Unicode cho Windows Console
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Thêm đường dẫn dự án
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        hn_path = "dataset_hanoi.parquet"
        hcm_path = "dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "../dataset_hanoi.parquet"
        hcm_path = "../dataset_hcmc.parquet"
        
    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        hn_path = "data/dataset_hanoi.parquet"
        hcm_path = "data/dataset_hcmc.parquet"

    if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu {hn_path} hoặc {hcm_path}.")
        sys.exit(1)
        
    df_hn = pd.read_parquet(hn_path)
    df_hn["city"] = "hanoi"
    df_hcm = pd.read_parquet(hcm_path)
    df_hcm["city"] = "hcmc"
    
    df_combined = pd.concat([df_hn, df_hcm], ignore_index=True)
    df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    return df_combined

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch

def main():
    print("🚀 BẮT ĐẦU PHÂN TÍCH LỖI MÔ HÌNH TFT CHO CẢ 6 CHẤT KHÍ...")
    os.makedirs("result", exist_ok=True)
    
    # 1. Đọc dữ liệu
    df = load_and_combine_datasets()
    
    t_min = df["timestamp"].min()
    df["time_idx"] = (df["timestamp"] - t_min).dt.total_seconds() // 3600
    df["time_idx"] = df["time_idx"].astype(int)
    
    validation_cutoff = int((pd.to_datetime("2025-10-01 00:00:00") - t_min).total_seconds() // 3600)
    test_start_idx = validation_cutoff - 72
    df = df[df.time_idx >= test_start_idx].reset_index(drop=True)
    
    # Ép kiểu dữ liệu phân loại
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
            
    # Shift vệ tinh 24h
    satellite_features = [
        "s5p_no2", "s5p_no2_cf", "s5p_so2", "s5p_so2_cf", "s5p_co", "s5p_co_cf", 
        "s5p_o3", "s5p_o3_cf", "s5p_aai", "s5p_aai_cf", "s5p_aod", "s5p_aod_cf", 
        "s5p_days_since_obs", "s5p_wind_alignment"
    ]
    cols_to_shift = [col for col in satellite_features if col in df.columns]
    if cols_to_shift:
        grouped = df.groupby(["city", "station_id"])
        df[cols_to_shift] = grouped[cols_to_shift].shift(24)
        df[cols_to_shift] = grouped[cols_to_shift].ffill().bfill()
        
    # 2. Khôi phục TimeSeriesDataSet
    params_path = "models/training_dataset_params.pkl"
    if not os.path.exists(params_path):
        params_path = "../models/training_dataset_params.pkl"
    if not os.path.exists(params_path):
        params_path = "training_dataset_params.pkl"
        
    with open(params_path, "rb") as f:
        dataset_params = pickle.load(f)
        
    test_dataset = TimeSeriesDataSet.from_parameters(
        dataset_params, df, min_prediction_idx=validation_cutoff + 1, stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=512, num_workers=0)
    
    # 3. Load checkpoint
    checkpoint_path = "models/tft-best-model-epoch=01-val_loss=14.4022.ckpt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "../models/tft-best-model-epoch=01-val_loss=14.4022.ckpt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "tft-best-model-epoch=01-val_loss=14.4022.ckpt"
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📖 Đang tải mô hình TFT từ checkpoint: {checkpoint_path}...")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    
    # 4. Trích xuất metadata khớp thời gian
    test_timestamps = []
    test_cities = []
    test_stations = []
    test_weather_data = []
    
    df_index_test = test_dataset.decoded_index
    for _, row in df_index_test.iterrows():
        pred_time_idx = row["time_idx_first_prediction"] + 11 # t+12
        meta_row = df[(df["station_id"] == row["station_id"]) & (df["time_idx"] == pred_time_idx)]
        if len(meta_row) > 0:
            meta = meta_row.iloc[0]
            test_timestamps.append(meta["timestamp"])
            test_cities.append(meta["city"])
            test_stations.append(meta["station_id"])
            test_weather_data.append((
                float(meta["temp_c"]) if "temp_c" in df.columns else 0.0,
                float(meta["wind_speed_ms"]) if "wind_speed_ms" in df.columns else 0.0,
                float(meta["dew_point_spread"]) if "dew_point_spread" in df.columns else 0.0,
                int(float(meta["is_peak_traffic"])) if "is_peak_traffic" in df.columns else 0
            ))
            
    # 5. Chạy dự báo
    all_preds = [[] for _ in range(6)]
    all_actuals = [[] for _ in range(6)]
    
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Đang dự báo"):
            x, y = batch
            x_device = to_device(x, device)
            out = model(x_device)
            pred = model.to_prediction(out)
            
            targets_actual = y[0] if isinstance(y, tuple) else y
            for i in range(6):
                all_preds[i].append(pred[i].cpu())
                all_actuals[i].append(targets_actual[i].cpu())
                
    actuals_dict = {}
    predictions_dict = {}
    
    n_preds = len(torch.cat(all_preds[0], dim=0))
    test_timestamps = test_timestamps[:n_preds]
    test_cities = test_cities[:n_preds]
    test_stations = test_stations[:n_preds]
    test_weather_data = test_weather_data[:n_preds]
    
    df_meta_by_gas = {}
    
    # 6. Gom dữ liệu kết quả để phân tích sai số cho cả 6 khí
    for i, target in enumerate(TARGET_COLUMNS):
        pred_vals = torch.cat(all_preds[i], dim=0).float().numpy()
        actual_vals = torch.cat(all_actuals[i], dim=0).float().numpy()
        
        act_gas = actual_vals[:, 11] # mốc t+12
        pred_gas = pred_vals[:, 11]
        pred_gas = np.clip(pred_gas, 0.0, None)
        
        actuals_dict[target] = act_gas
        predictions_dict[target] = pred_gas
        
        df_gas_meta = pd.DataFrame({
            "timestamp": test_timestamps,
            "city": test_cities,
            "station_id": test_stations,
            "temp_c": [w[0] for w in test_weather_data],
            "wind_speed_ms": [w[1] for w in test_weather_data],
            "dew_point_spread": [w[2] for w in test_weather_data],
            "is_peak_traffic": [w[3] for w in test_weather_data],
            f"{target}_t+12": act_gas,
            "pred_t12": pred_gas,
            "residual_t12": act_gas - pred_gas
        })
        df_meta_by_gas[target] = df_gas_meta
        
    # 7. Vẽ biểu đồ tán xạ gộp 2x3 actual vs predicted
    print("🎨 Đang vẽ biểu đồ gộp 2x3 cho TFT...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGET_COLUMNS):
        disp_name = TARGET_DISPLAY_NAMES[target]
        ax = axes[idx]
        
        act = actuals_dict[target]
        pred = predictions_dict[target]
        
        colors = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
        sns.scatterplot(x=act, y=pred, ax=ax, alpha=0.3, color=colors[idx], edgecolor=None)
        
        min_val = min(act.min(), pred.min())
        max_val = max(act.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
        
        mae = mean_absolute_error(act, pred)
        r2 = r2_score(act, pred)
        
        ax.text(0.05, 0.88, f"MAE: {mae:.2f}\nR²: {r2:.3f}", transform=ax.transAxes, 
                fontsize=10, fontweight="bold", bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2"))
        
        ax.set_title(f"{disp_name} (t+12) - Actual vs Predicted", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Value", fontsize=9)
        ax.set_ylabel("Predicted Value", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle("TFT (Temporal Fusion Transformer): Actual vs Predicted Scatter Plots (6 Gases at t+12)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plot_path = "result/tft_actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅ Đã lưu biểu đồ tại: {plot_path}")
    
    # 8. Viết báo cáo Markdown chi tiết cho CẢ 6 KHÍ
    report_path = "result/tft_error_analysis.md"
    print(f"✍️ Đang xuất báo cáo phân tích lỗi TFT cho cả 6 khí tại: {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO PHÂN TÍCH LỖI DỰ BÁO ĐỘC LẬP: MÔ HÌNH TFT (6 CHẤT KHÍ)\n")
        f.write("*Phân tích chuyên sâu các trường hợp sai lệch tiêu biểu của Temporal Fusion Transformer ở mốc t+12 trên toàn bộ 6 khí*\n\n")
        f.write("Báo cáo này chứa các thông số đánh giá chi tiết lỗi dự báo của mô hình mạng học sâu **Temporal Fusion Transformer (TFT)** đối với 6 chất khí: **PM2.5, PM10, NO₂, SO₂, CO, O₃**.\n\n")
        f.write(f"Biểu đồ tán xạ gộp 2x3 của mô hình được lưu tại: [tft_actual_vs_predicted.png](file:///c:/Users/Admin/Documents/AI%20Chess%20GPT/result/tft_actual_vs_predicted.png)\n\n")
        f.write("--- \n\n")
        
        for target in TARGET_COLUMNS:
            disp_name = TARGET_DISPLAY_NAMES[target]
            f.write(f"## 📊 PHÂN TÍCH CHẤT KHÍ: {disp_name}\n\n")
            
            df_gas = df_meta_by_gas[target]
            
            # Lấy ví dụ Underprediction tệ nhất (Thực tế >> Dự đoán)
            worst_under_df = df_gas.sort_values(by="residual_t12", ascending=False).head(2)
            # Lấy ví dụ Overprediction tệ nhất (Dự đoán >> Thực tế)
            worst_over_df = df_gas.sort_values(by="residual_t12", ascending=True).head(2)
            
            # A. Underprediction Table
            f.write("### Kịch bản A: Bỏ lỡ đỉnh ô nhiễm cực đại (Underprediction - Thực tế >> Dự đoán)\n")
            if len(worst_under_df) > 0:
                f.write("| Thời điểm | Thành phố | Trạm | Gió (m/s) | Lệch điểm sương (°C) | Thực tế (t+12) | Dự đoán (t+12) | Lệch phần dư |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for _, row in worst_under_df.iterrows():
                    act_val = row[f"{target}_t+12"]
                    pred_val = row["pred_t12"]
                    res_val = row["residual_t12"]
                    f.write(f"| {row['timestamp']} | {row['city']} | {row['station_id']} | {row['wind_speed_ms']:.2f} | {row['dew_point_spread']:.1f} | **{act_val:.2f}** | **{pred_val:.2f}** | **+{res_val:.2f}** |\n")
            else:
                f.write("*Không có dữ liệu*\n")
            f.write("\n")
            
            # B. Overprediction Table
            f.write("### Kịch bản B: Báo động giả ô nhiễm (Overprediction - Dự đoán >> Thực tế)\n")
            if len(worst_over_df) > 0:
                f.write("| Thời điểm | Thành phố | Trạm | Gió (m/s) | Lệch điểm sương (°C) | Thực tế (t+12) | Dự đoán (t+12) | Lệch phần dư |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for _, row in worst_over_df.iterrows():
                    act_val = row[f"{target}_t+12"]
                    pred_val = row["pred_t12"]
                    res_val = row["residual_t12"]
                    f.write(f"| {row['timestamp']} | {row['city']} | {row['station_id']} | {row['wind_speed_ms']:.2f} | {row['dew_point_spread']:.1f} | **{act_val:.2f}** | **{pred_val:.2f}** | **{res_val:.2f}** |\n")
            else:
                f.write("*Không có dữ liệu*\n")
            f.write("\n")
            
            # C. Hypotheses
            f.write("### 💡 Giả thuyết nguyên nhân cụ thể cho " + disp_name + ":\n")
            if target in ["pm25_obs", "pm10_obs"]:
                f.write("* **Hiện tượng nghịch nhiệt sương mù:** Vào mùa đông tại Hà Nội, khi gió lặng ($<1.0 m/s$) và lệch điểm sương nhỏ (sương mù ẩm cao), bụi mịn bị giam giữ sát mặt đất tạo ra các đỉnh ô nhiễm cực đoan. TFT có xu hướng học quy luật chu kỳ trơn tru và tối ưu hóa hàm mất mát trên toàn bộ phân phối nên bị kẹt ở mức trung bình, dẫn đến hiện tượng underprediction đỉnh.\n")
                f.write("* **Quán tính bộ nhớ LSTM Encoder:** Trạng thái ẩn dài hạn của LSTM Encoder lưu trữ giá trị ô nhiễm cao từ quá khứ quá mạnh, khiến mô hình phản ứng chậm chạp khi thời tiết dọn sạch khí quyển đột ngột (do mưa rào hoặc gió mùa), gây ra hiện tượng báo động giả (overprediction).\n")
            elif target == "co_pseudo":
                f.write("* **Đỉnh xung phát thải giao thông đô thị:** CO phụ thuộc trực tiếp vào lưu lượng phương tiện. Khi có ùn tắc giao thông cục bộ cực độ xung quanh trạm đo, nồng độ CO tăng vọt đột biến. TFT thiếu các đặc trưng lưu lượng xe cộ động thời gian thực (chỉ có biến chỉ thị tĩnh `is_peak_traffic`), cộng thêm cơ chế Variable Selection Network (VSN) của TFT có thể đã hạ thấp trọng số của biến giao thông tĩnh này khi dự báo dài hạn, dẫn đến trượt đỉnh CO.\n")
            elif target == "no2_pseudo":
                f.write("* **Hạn chế của dữ liệu vệ tinh Sentinel-5P:** NO₂ là chất khí phân hủy nhanh và biến động mạnh theo giờ. Dữ liệu cột mật độ NO₂ từ vệ tinh Sentinel-5P chỉ được quét 1 lần/ngày. Tại thời điểm sáng sớm (5:00 sáng), mô hình phải dùng dữ liệu quét cũ từ trưa hôm trước, dẫn tới việc bỏ lỡ sự tích tụ NO₂ từ khí thải phương tiện giao thông ban đêm và sáng sớm.\n")
            elif target == "o3_pseudo":
                f.write("* **Phản ứng hóa học quang hóa phi tuyến phức tạp:** Ozone được sinh ra từ phản ứng hóa học giữa NOₓ và VOCs dưới bức xạ tia cực tím và nhiệt độ cao. TFT không có thông số dự báo bức xạ mặt trời tương lai làm đặc trưng dẫn đường (future inputs), khiến mô hình phải tự ngoại suy chu kỳ sinh ra O₃. Khi thời tiết thay đổi đột xuất (mưa âm u đột ngột làm triệt tiêu phản ứng quang hóa), mô hình vẫn dự đoán O₃ cao dựa trên chu kỳ lịch sử nắng nóng trước đó, tạo ra sai số overpredict lớn.\n")
            elif target == "so2_pseudo":
                f.write("* **Hướng gió thổi vệt khói công nghiệp:** SO₂ phát sinh cục bộ từ các KCN. Mô hình phụ thuộc cực kỳ lớn vào độ chính xác của dự báo hướng gió để quyết định xem vệt khí SO₂ có thổi về phía trạm quan trắc hay không. Nếu hướng gió thay đổi đột ngột hoặc dự báo hướng gió bị nhiễu, TFT sẽ dự báo sai hoàn toàn nồng độ SO₂.\n")
                
            f.write("\n---\n\n")
            
        f.write("## 💡 ĐỀ XUẤT CẢI TIẾN DÀNH RIÊNG CHO MÔ HÌNH TFT\n\n")
        f.write("1. **Bổ sung các đặc trưng tương lai biết trước (Known Future Covariates):** Tích hợp các biến dự báo thời tiết tương lai (như dự báo nhiệt độ, tốc độ gió, độ che phủ mây/bức xạ UV dự kiến) vào nhánh `known_recurrent_covariates` của TFT thay vì chỉ dùng các giá trị quá khứ làm đặc trưng tĩnh/trễ.\n")
        f.write("2. **Sử dụng Loss Function tối ưu hóa phân vị (Quantile Loss):** Tận dụng tối đa khả năng dự báo phân vị của TFT để phân tích độ bất định (uncertainty intervals). Cần tập trung đánh giá phân vị cao (như $p_{0.9}$) để chuẩn bị cho các kịch bản đỉnh ô nhiễm cực đại, giảm thiểu lỗi Underprediction.\n")

    print(f"\n🎉 HOÀN THÀNH QUÁ TRÌNH PHÂN TÍCH LỖI CHO TFT!")
    print(f"📄 Báo cáo đã được lưu tại: {report_path}")

if __name__ == "__main__":
    main()
