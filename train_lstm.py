"""
train_lstm.py
=====================================================================
Kịch bản huấn luyện và đánh giá mô hình Stacked BiLSTM cho CẢ 6 KHÍ
trên chuỗi dự báo 24 giờ tiếp theo (t+1 đến t+24).

ĐẢM BẢO SO SÁNH CÔNG BẰNG:
- Load dữ liệu Hà Nội và TP.HCM giống tft.py
- Phân chia tập dữ liệu theo thời gian (Train < 2025-01-01, Val 2025-01-01 -> 2025-10-01, Test >= 2025-10-01) giống hệt LightGBM/TFT.
- Chuẩn hóa đúng phương pháp khoa học (Fit StandardScaler chỉ trên tập Train để tránh Data Leakage).
- Dự báo Multi-output song song 24 giờ tiếp theo (Horizon = 24) cho cả 6 chất khí.
=====================================================================
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fix Unicode cho Windows Console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, LayerNormalization, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ══════════════════════════════════════════════════════════════════
# 1. SIÊU THAM SỐ & CONFIG
# ══════════════════════════════════════════════════════════════════
WINDOW_SIZE   = 24      # Nhìn lại 24 tiếng quá khứ (t-23 đến t)
HORIZON       = 24      # Dự báo 24 tiếng tiếp theo (t+1 đến t+24)
LSTM_UNITS_1  = 128     # Hidden units tầng LSTM thứ nhất
LSTM_UNITS_2  = 64      # Hidden units tầng LSTM thứ hai
DROPOUT_RATE  = 0.2
BATCH_SIZE    = 512
EPOCHS        = 10
LEARNING_RATE = 1e-3
SEED          = 42

TARGETS = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
TARGET_DISPLAY_NAMES = {
    "pm25_obs": "PM2.5",
    "pm10_obs": "PM10",
    "no2_pseudo": "NO₂",
    "so2_pseudo": "SO₂",
    "co_pseudo": "CO",
    "o3_pseudo": "O₃",
}

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ══════════════════════════════════════════════════════════════════
# 2. TẢI DỮ LIỆU (Giống hệt tft.py & train_lightgbm.py)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 1 — Tải dữ liệu (Hanoi & HCMC Combined)")
print("="*65)

# Đường dẫn Kaggle theo tft.py
hn_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hanoi.parquet"
hcm_path = "/kaggle/input/datasets/nguynquclc/adhfahr/dataset_hcmc.parquet"

# Fallback nếu chạy test local
if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
    hn_path = "data/dataset_hanoi.parquet"
    hcm_path = "data/dataset_hcmc.parquet"
    
if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
    hn_path = "dataset_hanoi.parquet"
    hcm_path = "dataset_hcmc.parquet"

if not os.path.exists(hn_path) or not os.path.exists(hcm_path):
    print(f"❌ LỖI: Không tìm thấy file dữ liệu tại '{hn_path}' hoặc '{hcm_path}'.")
    sys.exit(1)

print(f"📖 Đang đọc dữ liệu Hà Nội từ: {hn_path}...")
df_hn = pd.read_parquet(hn_path)
df_hn["city"] = "hanoi"

print(f"📖 Đang đọc dữ liệu TP.HCM từ: {hcm_path}...")
df_hcm = pd.read_parquet(hcm_path)
df_hcm["city"] = "hcmc"

df = pd.concat([df_hn, df_hcm], ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["city", "station_id", "timestamp"]).reset_index(drop=True)
print(f"✅ Tải thành công! Tổng số dòng gộp: {len(df):,}")

# ══════════════════════════════════════════════════════════════════
# 3. CHỌN FEATURES & XỬ LÝ NULL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 2 — Chọn features & tiền xử lý")
print("="*65)

DROP_COLS = [
    "timestamp", "station_id", "station_name", "data_source",
    "city", "pm25_source", "pm10_source",
    "no2_source", "so2_source", "co_source", "o3_source",
]

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
print(f"  Số features sử dụng: {len(FEATURE_COLS)}")

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
df[TARGETS]      = df[TARGETS].fillna(0)

# ══════════════════════════════════════════════════════════════════
# 4. CHIA TÁCH THEO THỜI GIAN (Giống hệt LightGBM/XGBoost)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
# ══════════════════════════════════════════════════════════════════
# 4. CHUẨN HÓA DỮ LIỆU ĐÚNG PHƯƠNG PHÁP KHOA HỌC (Không Data Leakage)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 4 — Chuẩn hóa dữ liệu (Chỉ Fit trên tập Train)")
print("="*65)

# Bản sao tập Train chưa scale để fit bộ scaler
df_train_raw = df[df["timestamp"] < pd.to_datetime("2025-01-01 00:00:00")].copy()

scaler_y = StandardScaler()
scaler_y.fit(df_train_raw[TARGETS])

# Tách biệt đặc trưng không phải target để tránh chuẩn hóa chồng lấn (Double Scaling)
NON_TARGET_FEATURES = [c for c in FEATURE_COLS if c not in TARGETS]
scaler_X = StandardScaler()
scaler_X.fit(df_train_raw[NON_TARGET_FEATURES])

# Giải phóng bộ nhớ bản sao tập Train thô
del df_train_raw
gc.collect()

# Transform lên toàn bộ dataframe (mỗi cột chỉ chuẩn hóa đúng 1 lần)
df[NON_TARGET_FEATURES] = scaler_X.transform(df[NON_TARGET_FEATURES])
df[TARGETS]             = scaler_y.transform(df[TARGETS])

print("  ✅ Đã hoàn thành chuẩn hóa sòng phẳng.")

# ══════════════════════════════════════════════════════════════════
# 5. TẠO SLIDING WINDOW CÓ HORIZON = 24 CHO TỪNG TRẠM
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 5 — Tạo sliding window (Horizon = 24h)")
print("="*65)

def build_sequences(df_subset, split_name="train"):
    station_data = []
    indices = []
    city_names = []
    s_idx = 0
    
    t_train_cutoff = pd.to_datetime("2025-01-01 00:00:00")
    t_test_cutoff = pd.to_datetime("2025-10-01 00:00:00")
    
    for (city, station_id), grp in df_subset.groupby(["city", "station_id"], sort=False):
        n_samples = len(grp)
        # Cần tối thiểu WINDOW_SIZE + HORIZON để tạo được 1 sequence dự báo đầy đủ 24h
        if n_samples > WINDOW_SIZE + HORIZON:
            X_vals = grp[FEATURE_COLS].values.astype(np.float32)
            y_vals = grp[TARGETS].values.astype(np.float32)
            timestamps = grp["timestamp"].values
            
            n_seq = n_samples - WINDOW_SIZE - HORIZON + 1
            station_added = False
            for i in range(n_seq):
                # Kiểm tra mốc t+1 (bước dự báo đầu tiên) để phân chia tập dữ liệu
                t_plus_1 = pd.to_datetime(timestamps[i + WINDOW_SIZE])
                
                if split_name == "train" and t_plus_1 < t_train_cutoff:
                    indices.append((s_idx, i))
                    city_names.append(city)
                    station_added = True
                elif split_name == "val" and t_train_cutoff <= t_plus_1 < t_test_cutoff:
                    indices.append((s_idx, i))
                    city_names.append(city)
                    station_added = True
                elif split_name == "test" and t_plus_1 >= t_test_cutoff:
                    indices.append((s_idx, i))
                    city_names.append(city)
                    station_added = True
            
            if station_added:
                station_data.append((X_vals, y_vals))
                s_idx += 1
                
    return station_data, indices, city_names

print("⏳ Đang chuẩn bị sequence cho tập Train...")
train_station_data, train_indices, _ = build_sequences(df, "train")
print("⏳ Đang chuẩn bị sequence cho tập Val...")
val_station_data, val_indices, _ = build_sequences(df, "val")
print("⏳ Đang chuẩn bị sequence cho tập Test...")
test_station_data, test_indices, test_cities = build_sequences(df, "test")

# Giải phóng df gốc để tiết kiệm bộ nhớ tối đa trước khi huấn luyện
del df
gc.collect()

print(f"  Train sequences: {len(train_indices):,}")
print(f"  Val sequences:   {len(val_indices):,}")
print(f"  Test sequences:  {len(test_indices):,}")

# ══════════════════════════════════════════════════════════════════
# 7. XÂY DỰNG DATA PIPELINE BẰNG TF.DATA
# ══════════════════════════════════════════════════════════════════
n_features = len(FEATURE_COLS)
n_targets = len(TARGETS)

def make_tf_dataset(station_data, indices, shuffle=False):
    def gen():
        for s_idx, i in indices:
            X_vals, y_vals = station_data[s_idx]
            # X_seq: 24h quá khứ (WINDOW_SIZE)
            # y_seq: 24h tiếp theo (HORIZON)
            X_seq = X_vals[i : i + WINDOW_SIZE]
            y_seq = y_vals[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON]
            yield X_seq, y_seq

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SIZE, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(HORIZON, n_targets),      dtype=tf.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_train = make_tf_dataset(train_station_data, train_indices, shuffle=True)
ds_val   = make_tf_dataset(val_station_data, val_indices)
ds_test  = make_tf_dataset(test_station_data, test_indices)

# Materialize tập Test numpy để tính toán metrics offline nhanh chóng
print("⏳ Đang thu thập tập Test numpy để đánh giá...")
X_test_np = []
y_test_np = []
for x_b, y_b in ds_test:
    X_test_np.append(x_b.numpy())
    y_test_np.append(y_b.numpy())
X_test_np = np.concatenate(X_test_np, axis=0)
y_test_np = np.concatenate(y_test_np, axis=0)

# Reset lại generator test
ds_test = make_tf_dataset(test_station_data, test_indices)

# ══════════════════════════════════════════════════════════════════
# 8. XÂY DỰNG KIẾN TRÚC MÔ HÌNH STACKED BiLSTM CHO CHUỖI 24H
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 6 — Xây dựng mô hình Stacked BiLSTM")
print("="*65)

# Phân tích tham số dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá mô hình đã lưu mà không huấn luyện lại")
args, unknown = parser.parse_known_args()

# Thiết lập chiến lược phân tán đa GPU nếu phát hiện nhiều GPU (ví dụ song song 2x T4 trên Kaggle)
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    print(f"🚀 Phát hiện {len(gpus)} GPUs! Kích hoạt MirroredStrategy để huấn luyện song song đa GPU.")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("🚀 Sử dụng Single-GPU hoặc CPU mặc định.")
    strategy = tf.distribute.get_strategy()

if args.eval_only:
    model_path = "best_lstm_24h_model.keras"
    if not os.path.exists(model_path):
        model_path = "lstm_airquality_24h_model.keras"
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy file model để load ({model_path}).")
        sys.exit(1)
    
    print(f"🚀 Chế độ chỉ đánh giá (Eval-only): Đang tải mô hình từ {model_path}...")
    with strategy.scope():
        model = tf.keras.models.load_model(model_path)
    
    model.summary()
else:
    with strategy.scope():
        inputs = Input(shape=(WINDOW_SIZE, n_features), name="input")

        # Tầng BiLSTM 1 - Trả về sequence để xếp tầng tiếp theo
        x = Bidirectional(
            LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE),
            name="bi_lstm_1"
        )(inputs)
        x = LayerNormalization(name="layer_norm_1")(x)

        # Tầng BiLSTM 2 - Trả về vector tổng hợp cuối cùng
        x = Bidirectional(
            LSTM(LSTM_UNITS_2, return_sequences=False, dropout=DROPOUT_RATE),
            name="bi_lstm_2"
        )(x)
        x = LayerNormalization(name="layer_norm_2")(x)

        # Mạng nơ-ron Dense kết nối
        x = Dense(128, activation="relu", name="dense_1")(x)
        x = Dropout(DROPOUT_RATE, name="dropout_out")(x)
        x = Dense(64, activation="relu", name="dense_2")(x)

        # Nút ra tuyến tính tương đương 24h x 6 targets
        x = Dense(HORIZON * n_targets, activation="linear", name="dense_output")(x)

        # Reshape kết quả đầu ra về dạng (Horizon=24, Targets=6) giống cấu trúc mong đợi
        outputs = Reshape((HORIZON, n_targets), name="output")(x)

        model = Model(inputs, outputs, name="StackedBiLSTM_24h_Forecast")
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="mse",
            metrics=["mae"]
        )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            "best_lstm_24h_model.keras",
            monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

# ══════════════════════════════════════════════════════════════════
# 10. ĐÁNH GIÁ CHUẨN XÁC TRÊN TẬP TEST (Inverse Scaled)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BƯỚC 8 — Đánh giá sòng phẳng trên tập Test")
print("="*65)

y_pred_scaled = model.predict(X_test_np, batch_size=BATCH_SIZE, verbose=0)

# Khôi phục giá trị gốc của 6 khí từ scaler_y
# scaler_y có input dạng (N, 6). Để dùng inverse_transform, ta phải flatten chiều 24h
# y_pred_scaled shape: (N, 24, 6) -> reshape thành (N * 24, 6)
N_test = len(y_pred_scaled)
y_pred_flat = y_pred_scaled.reshape(N_test * HORIZON, n_targets)
y_test_flat = y_test_np.reshape(N_test * HORIZON, n_targets)

y_pred_orig_flat = scaler_y.inverse_transform(y_pred_flat)
y_test_orig_flat = scaler_y.inverse_transform(y_test_flat)

# Đưa trở lại dạng 3D gốc
y_pred_orig = y_pred_orig_flat.reshape(N_test, HORIZON, n_targets)
y_test_orig = y_test_orig_flat.reshape(N_test, HORIZON, n_targets)

print(f"  → Số mẫu Test sử dụng để đánh giá: {len(y_pred_orig):,}")

# Tính toán sai số trung bình (Mean over all 24 horizons) cho từng khí
lstm_results = {}
for i, target in enumerate(TARGETS):
    disp_name = TARGET_DISPLAY_NAMES[target]
    
    # Lấy nhãn thực tế & dự đoán của khí tương ứng trên toàn bộ tập Test
    act_gas = y_test_orig[:, :, i]
    pred_gas = y_pred_orig[:, :, i]
    
    mae = mean_absolute_error(act_gas, pred_gas)
    rmse = np.sqrt(mean_squared_error(act_gas, pred_gas))
    r2 = r2_score(act_gas.flatten(), pred_gas.flatten())
    
    lstm_results[disp_name] = {"mae": mae, "rmse": rmse, "r2": r2}

# In bảng kết quả LSTM
print("\n" + "="*55)
print(f"📊 BẢNG TỔNG HỢP HIỆU NĂNG LSTM TRÊN TẬP TEST 10% (HN + HCM)")
print("="*55)
print(f"{'Chất khí':<10} | {'LSTM MAE':<12} {'LSTM RMSE':<12} {'LSTM R²':<10}")
print("-" * 55)
for gas in lstm_results.keys():
    lst_mae = lstm_results[gas]["mae"]
    lst_rmse = lstm_results[gas]["rmse"]
    lst_r2 = lstm_results[gas]["r2"]
    print(f"{gas:<10} | {lst_mae:<12.2f} {lst_rmse:<12.2f} {lst_r2:<10.3f}")
print("="*55)

# Vẽ biểu đồ MAE của Stacked BiLSTM
gas_names = list(lstm_results.keys())
lstm_maes = [lstm_results[g]["mae"] for g in gas_names]

x = np.arange(len(gas_names))

fig, ax = plt.subplots(figsize=(10, 5))
rects = ax.bar(x, lstm_maes, 0.5, label='Stacked BiLSTM', color='royalblue')

ax.set_ylabel('MAE (μg/m³)')
ax.set_title('Sai số MAE của Stacked BiLSTM trên tập Test 10%')
ax.set_xticks(x)
ax.set_xticklabels(gas_names)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_mae_on_test_set.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Đã lưu biểu đồ: lstm_mae_on_test_set.png")

# ══════════════════════════════════════════════════════════════════
# 11. LƯU MÔ HÌNH & KẾT THÚC
# ══════════════════════════════════════════════════════════════════
model.save("lstm_airquality_24h_model.keras")
print("\n✅ Hoàn tất lưu mô hình và kết thúc chạy thử nghiệm!")
