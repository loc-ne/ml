import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch

import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
import pytorch_forecasting
torch.serialization.add_safe_globals([pytorch_forecasting.data.encoders.MultiNormalizer])
print("🚀 Khởi động môi trường TFT...")

# Đổi đường dẫn dataset
DATA_DIR = "./data" 

df_hanoi = pd.read_parquet(f"/kaggle/input/datasets/arisene/train-final/dataset_hanoi.parquet")
df_hcmc = pd.read_parquet(f"/kaggle/input/datasets/arisene/train-final/dataset_hcmc.parquet")


df = pd.concat([df_hanoi, df_hcmc], ignore_index=True)
print(f"✅ Đã tải dữ liệu. Tổng số dòng: {len(df):,}")


df["timestamp"] = pd.to_datetime(df["timestamp"])
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
df["time_idx"] = df["time_idx"].astype(int)

# Các cột phân loại bắt buộc phải là kiểu str/category
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

print("✅ Đã hoàn tất ép kiểu dữ liệu và tạo time_idx.")

# =====================================================================
# BƯỚC 3: ĐỊNH NGHĨA TIMESERIES DATASET
# =====================================================================
max_encoder_length = 72  # Nhìn lại 3 ngày
max_prediction_length = 24 # Dự báo 24h tới

# Chuẩn Production: Phân chia Train - Val - Test đồng bộ với XGBoost/LightGBM
t_min = df["timestamp"].min()
training_cutoff = int((pd.to_datetime("2025-01-01 00:00:00") - t_min).total_seconds() // 3600)
validation_cutoff = int((pd.to_datetime("2025-10-01 00:00:00") - t_min).total_seconds() // 3600)

print(f"📊 Cutoff points - Train: <= {training_cutoff}, Val: <= {validation_cutoff}, Test: > {validation_cutoff}")

# Lọc các biến thực tế có tồn tại trong df
static_reals = [c for c in [
    "elevation_m", "population_density", "dist_to_industrial_km", 
    "dist_to_center_km", "land_use_built_pct", "angle_to_industrial_deg"
] if c in df.columns]

known_cats = [c for c in categorical_cols if c not in ["city", "station_id", "data_source"] and not c.endswith("_source") and c in df.columns]
unknown_cats = [c for c in categorical_cols if (c.endswith("_source") or c == "data_source") and c in df.columns]

# ĐƯA THỜI TIẾT & ĐỘNG LỰC HỌC VÀO KNOWN_REALS (Khí tượng tương lai có thể dự báo trước rất chuẩn bằng API)
weather_forecast_features = [
    # Thời tiết cơ bản
    "temp_c", "humidity_pct", "wind_speed_ms", "precipitation_mm", "pressure_hpa", 
    "dewpoint_c", "cloud_cover_pct", "shortwave_rad", "wind_gust_ms", "boundary_layer_h",
    # Động lực học không khí
    "wind_dir_sin", "wind_dir_cos", "ventilation_index", "stagnation_index", 
    "industrial_wind_factor", "dew_point_spread", "sunlight_proxy", "air_density_proxy"
]

known_reals = [c for c in [
    "time_idx", "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"
] + weather_forecast_features if c in df.columns]

# Targets chính thức: Bụi đo trực tiếp + 4 khí từ Pseudo (đã gộp Real + Downscaled)
target_columns = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]

# Các biến vệ tinh (không thể biết trước tương lai) và target lịch sử đưa vào unknown_reals
satellite_features = [
    "s5p_no2", "s5p_no2_cf", "s5p_so2", "s5p_so2_cf", "s5p_co", "s5p_co_cf", 
    "s5p_o3", "s5p_o3_cf", "s5p_aai", "s5p_aai_cf", "s5p_aod", "s5p_aod_cf", 
    "s5p_days_since_obs", "s5p_wind_alignment"
]

# Xử lý trễ dữ liệu vệ tinh thực tế (Satellite Data Latency - Trễ 24h):
# Dịch chuyển dữ liệu vệ tinh đi 24h về tương lai để mô hình huấn luyện dựa trên dữ liệu thực tế trễ.
print("⏳ Đang xử lý độ trễ thực tế 24h của dữ liệu vệ tinh S5P...")
for col in satellite_features:
    if col in df.columns:
        df[col] = df.groupby(["city", "station_id"])[col].shift(24)
        df[col] = df.groupby(["city", "station_id"])[col].ffill().bfill()

unknown_reals = [c for c in satellite_features + target_columns if c in df.columns]

training_dataset = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=target_columns, # Multi-target đã lấp đầy hoàn hảo
    group_ids=["city", "station_id"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    
    static_categoricals=["city", "station_id"],
    static_reals=static_reals,
    
    time_varying_known_categoricals=known_cats,
    time_varying_known_reals=known_reals,
    
    time_varying_unknown_categoricals=unknown_cats,
    time_varying_unknown_reals=unknown_reals,
    
    allow_missing_timesteps=True, 
    
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, 
    df[lambda x: x.time_idx <= validation_cutoff], 
    min_prediction_idx=training_cutoff + 1, 
    stop_randomization=True
)

test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, 
    df, 
    min_prediction_idx=validation_cutoff + 1, 
    stop_randomization=True
)

print(f"✅ Đã tạo Dataset. Train size: {len(training_dataset):,}")

# =====================================================================
# BƯỚC 4: KHỞI TẠO DATALOADERS & MODEL
# =====================================================================
# CẤU HÌNH CHO GPU T4 (15GB VRAM) trên Kaggle:
batch_size = 512 
train_dataloader = training_dataset.to_dataloader(
    train=True, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
)
val_dataloader = validation_dataset.to_dataloader(
    train=False, batch_size=batch_size * 2, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
)
test_dataloader = test_dataset.to_dataloader(
    train=False, batch_size=batch_size * 2, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
)

# Tự động hóa trọng số MultiLoss chuẩn Production (Inverse Variance Weighting):
# Tính toán nghịch đảo độ lệch chuẩn (1 / std) của từng khí để cân bằng đóng góp loss của các khí có thang đo khác nhau.
train_df_subset = df[df.time_idx <= training_cutoff]
stds = []
for target in target_columns:
    std_val = train_df_subset[target].std()
    stds.append(std_val if std_val > 0 else 1.0)

raw_weights = [1.0 / s for s in stds]
sum_w = sum(raw_weights)
loss_weights = [w * len(target_columns) / sum_w for w in raw_weights]
print(f"📊 Trọng số MultiLoss chuẩn Production (Inverse Variance): {loss_weights}")

tft_loss = MultiLoss(
    metrics=[QuantileLoss() for _ in range(6)], 
    weights=loss_weights                       
)

# Nâng cấp dung lượng mô hình để tăng sức mạnh biểu diễn phi tuyến tính
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.001,
    hidden_size=128,             # Nâng từ 64 lên 128
    lstm_layers=2,               
    attention_head_size=8,       # Nâng từ 4 lên 8 đầu attention
    dropout=0.30,                # Tăng dropout lên 0.30 chống overfitting
    hidden_continuous_size=64,   # Nâng từ 32 lên 64 cho đặc trưng liên tục
    loss=tft_loss,
    log_interval=10,
    reduce_on_plateau_patience=4,
)



tb_logger = TensorBoardLogger("lightning_logs", name="tft_air_quality")

# 2. Dừng sớm nếu Validation Loss không giảm sau 5 epochs (chống Overfitting)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5,
    verbose=True,
    mode="min"
)

# 3. Theo dõi tốc độ học (Learning Rate)
lr_logger = LearningRateMonitor(logging_interval='epoch')

# 4. Tự động lưu mô hình xuất sắc nhất (dùng để Infer)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="models",
    filename="tft-best-model-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min",
)

# 5. Callback tự động ghi lại lịch sử Loss qua từng Epoch ra file CSV
import csv
from lightning.pytorch.callbacks import Callback

class LossHistoryCallback(Callback):
    def __init__(self, filepath="models/loss_history.csv"):
        super().__init__()
        self.filepath = filepath
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        train_loss = metrics.get("train_loss_epoch") or metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        
        if train_loss is not None and val_loss is not None:
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item()
            })
            
            # Ghi đè file CSV để liên tục cập nhật lịch sử loss
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
                writer.writeheader()
                writer.writerows(self.history)

loss_history_callback = LossHistoryCallback()

import pickle
os.makedirs("models", exist_ok=True)
dataset_params = training_dataset.get_parameters()
params_path = "models/training_dataset_params.pkl"
with open(params_path, "wb") as f:
    pickle.dump(dataset_params, f)
print(f"💾 Đã lưu Dataset Parameters: {params_path}")
print("👉 Download file này ngay bây giờ để dùng cho Deploy!")

# =====================================================================
# BƯỚC 7: HUẤN LUYỆN
# =====================================================================
print("🚀 Bắt đầu huấn luyện...")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",      
    devices=1,               
    num_sanity_val_steps=2,
    gradient_clip_val=0.1,
    enable_progress_bar=True,
    logger=tb_logger,
    callbacks=[early_stop_callback, lr_logger, checkpoint_callback, loss_history_callback],
)

# Cấu hình huấn luyện từ đầu (Không sử dụng checkpoint cũ do thay đổi kiến trúc mô hình)
checkpoint_path = "models/tft-best-model-epoch=01-val_loss=14.4022.ckpt"

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    weights_only=False,
    ckpt_path=checkpoint_path 
)

# =====================================================================
# BƯỚC 8: ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
# =====================================================================
print("\n📝 Đang tải mô hình tốt nhất để đánh giá trên tập Test...")
best_model_path = checkpoint_callback.best_model_path
print(f"🏆 Đường dẫn mô hình tốt nhất: {best_model_path}")

if best_model_path and os.path.exists(best_model_path):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    # Thực hiện dự báo trên tập Test
    print("🔮 Đang chạy dự báo trên tập Test...")
    predictions = best_tft.predict(test_dataloader, return_y=True)
    
    # Tính toán MAE cho từng khí trên tập Test
    for i, target in enumerate(target_columns):
        disp_name = target.replace("_obs", "").replace("_pseudo", "").upper()
        
        # Lấy giá trị dự báo và thực tế tương ứng với target thứ i
        if isinstance(predictions, tuple) and (isinstance(predictions[0], list) or isinstance(predictions[0], tuple)):
            pred_vals = predictions[0][i]
            actual_vals = predictions[1][i]
        elif isinstance(predictions, tuple):
            pred_vals = predictions[0]
            actual_vals = predictions[1]
        else:
            pred_vals = predictions
            actual_vals = None
            
        if actual_vals is not None:
            mae = torch.mean(torch.abs(pred_vals.float() - actual_vals.float())).item()
            print(f"📌 MAE trên tập Test cho {disp_name}: {mae:.4f}")
else:
    print("⚠️ Không tìm thấy đường dẫn mô hình tốt nhất để chạy đánh giá.")

