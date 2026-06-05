"""
deploy/live_test.py
===================
Công cụ dự báo và đối chiếu LIVE chuỗi 24h gần nhất.
Tải tự động thời tiết thực tế từ Open-Meteo Weather API
và dữ liệu chất lượng không khí thực tế từ Open-Meteo Air Quality API (không cần API Key).
Sau đó chạy mô hình TFT và tính toán sai số trực tiếp so với thực tế hôm qua!

Hỗ trợ chạy mặc định chế độ "live" (96h gần nhất) hoặc một ngày cụ thể trong quá khứ gần.
"""

import os
import sys
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Fix Unicode cho Windows Console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from deploy.predictor import TFTPredictor
from deploy.data_pipeline import compute_static_features, get_s5p_features, fetch_tomtom_traffic
from deploy.feature_engineer import engineer_all_features

DEFAULT_CHECKPOINT = str(
    Path(__file__).parent.parent / "models" / "tft-best-model-epoch=01-val_loss=14.4022.ckpt"
)
DEFAULT_DATASET_PARAMS = str(
    Path(__file__).parent.parent / "models" / "training_dataset_params.pkl"
)

TARGET_COLUMNS = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
TARGET_DISPLAY_NAMES = {
    "pm25_obs": "PM2.5",
    "pm10_obs": "PM10",
    "no2_pseudo": "NO₂",
    "so2_pseudo": "SO₂",
    "co_pseudo": "CO",
    "o3_pseudo": "O₃",
}
TARGET_UNITS = {
    "pm25_obs": "μg/m³",
    "pm10_obs": "μg/m³",
    "no2_pseudo": "μg/m³",
    "so2_pseudo": "μg/m³",
    "co_pseudo": "μg/m³",
    "o3_pseudo": "μg/m³",
}


def fetch_live_weather(lat: float, lon: float, start_date: str = None, end_date: str = None, past_hours: int = 96) -> pd.DataFrame:
    """Tải thực tế thời tiết từ Open-Meteo Weather API."""
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
        "surface_pressure", "precipitation", "boundary_layer_height", "shortwave_radiation",
        "cloud_cover", "dewpoint_2m", "wind_gusts_10m"
    ]
    
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(hourly_vars),
        "timezone":        "Asia/Ho_Chi_Minh",
        "wind_speed_unit": "ms",
    }
    
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_hours"] = past_hours
        params["forecast_hours"] = 0
        
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Lỗi tải Open-Meteo Weather API: {response.text}")
        
    data = response.json()
    hourly = data["hourly"]
    
    df = pd.DataFrame({
        "timestamp":        pd.to_datetime(hourly["time"]),
        "temp_c":           hourly.get("temperature_2m"),
        "humidity_pct":     hourly.get("relative_humidity_2m"),
        "pressure_hpa":     hourly.get("surface_pressure"),
        "precipitation_mm": hourly.get("precipitation"),
        "dewpoint_c":       hourly.get("dewpoint_2m"),
        "shortwave_rad":    hourly.get("shortwave_radiation"),
        "wind_speed_ms":    hourly.get("wind_speed_10m"),
        "wind_dir_deg":     hourly.get("wind_direction_10m"),
        "boundary_layer_h": hourly.get("boundary_layer_height"),
        "wind_gust_ms":     hourly.get("wind_gusts_10m"),
        "cloud_cover_pct":  hourly.get("cloud_cover"),
    })
    
    # Clip và nội suy nhẹ
    clip_rules = {
        "temp_c": (-20, 50), "humidity_pct": (0, 100), "wind_speed_ms": (0, 50),
        "precipitation_mm": (0, 500), "boundary_layer_h": (0, 5000),
        "dewpoint_c": (-30, 50), "wind_gust_ms": (0, 80),
        "cloud_cover_pct": (0, 100), "shortwave_rad": (0, 1500),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    df = df.interpolate(method="linear", limit=3)
    return df


def fetch_live_aqi(lat: float, lon: float, start_date: str = None, end_date: str = None, past_hours: int = 96) -> pd.DataFrame:
    """Tải thực tế chất lượng không khí từ Open-Meteo Air Quality API."""
    if start_date and end_date:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone&start_date={start_date}&end_date={end_date}&timezone=Asia/Ho_Chi_Minh"
    else:
        past_days = int(np.ceil(past_hours / 24)) + 1
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone&past_days={past_days}&timezone=Asia/Ho_Chi_Minh"
    
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Lỗi tải Open-Meteo Air Quality API: {response.text}")
        
    data = response.json()
    hourly = data["hourly"]
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm25_obs": hourly["pm2_5"],
        "pm10_obs": hourly["pm10"],
        "no2_pseudo": hourly["nitrogen_dioxide"],
        "so2_pseudo": hourly["sulphur_dioxide"],
        "co_pseudo": hourly["carbon_monoxide"],
        "o3_pseudo": hourly["ozone"],
    })
    return df


def run_live_test(lat: float, lon: float, city: str, start_date_str: str, checkpoint_path: str, params_path: str, apply_calibration: bool = False):
    print(f"\n🌍 BẮT ĐẦU CHẠY LIVE VALIDATION TẠI TỌA ĐỘ ({lat}, {lon}) - THÀNH PHỐ: {city.upper()}")
    print(f"⚠️  BỘ HIỆU CHỈNH: {'BẬT' if apply_calibration else 'TẮT (Raw output)'}")
    print("═" * 80)
    
    # Phân tích ngày chạy
    is_live_mode = (start_date_str.lower() == "live")
    
    start_date_api = None
    end_date_api = None
    
    if not is_live_mode:
        try:
            start_dt = pd.to_datetime(start_date_str)
            end_dt = start_dt + timedelta(days=3) # 4 ngày liên tiếp = 96 giờ
            start_date_api = start_dt.strftime("%Y-%m-%d")
            end_date_api = end_dt.strftime("%Y-%m-%d")
            print(f"📅 Chế độ chọn ngày: Dự báo chuỗi 96 tiếng từ {start_date_api} đến {end_date_api}")
        except Exception as e:
            print(f"❌ Lỗi định dạng ngày nhập vào '{start_date_str}': {e}")
            print("👉 Định dạng đúng: 'YYYY-MM-DD' (Ví dụ: '2026-05-10')")
            return
    else:
        print("⚡ Chế độ LIVE: Tải tự động 96 giờ gần nhất tính từ thời điểm hiện tại.")

    # 1. Tải Weather thực tế
    print("⏳ Bước 1: Đang tải thời tiết thực tế từ Open-Meteo Weather API...")
    df_weather = fetch_live_weather(lat, lon, start_date=start_date_api, end_date=end_date_api, past_hours=96)
    print(f"   → Tải thành công {len(df_weather)} giờ khí tượng.")

    # 2. Tải AQI thực tế
    print("⏳ Bước 2: Đang tải dữ liệu chất lượng không khí thực tế từ Open-Meteo Air Quality API...")
    df_aqi = fetch_live_aqi(lat, lon, start_date=start_date_api, end_date=end_date_api, past_hours=96)
    print(f"   → Tải thành công {len(df_aqi)} giờ AQI thực tế.")

    # Gộp 2 DataFrame
    df_full = pd.merge(df_weather, df_aqi, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    
    # Cắt lấy đúng 96 dòng chuẩn đầu tiên/hoặc gần nhất
    if is_live_mode:
        df_full = df_full.tail(96).copy().reset_index(drop=True)
    else:
        df_full = df_full.head(96).copy().reset_index(drop=True)
        
    if len(df_full) < 96:
        print(f"❌ Không đủ 96 giờ khớp nhau (chỉ có {len(df_full)} giờ).")
        print("👉 Vui lòng thử chọn mốc ngày khác, hoặc kiểm tra kết nối mạng của bạn.")
        return
        
    df_full["time_idx"] = range(len(df_full))

    # 3. Điền thông tin trạm và nguồn dữ liệu để khớp Encoder
    df_full["station_id"] = "openaq_2161290" if city == "hcmc" else "openaq_7441"
    df_full["city"] = city
    df_full["data_source"] = "openaq"
    for src in ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]:
        df_full[src] = "real"

    # Thêm biến tĩnh GIS
    static = compute_static_features(lat, lon)
    for k, v in static.items():
        df_full[k] = v

    # Thêm biến vệ tinh S5P (Động cho khí, Tĩnh cho bụi)
    s5p_const = get_s5p_features(lat=lat, lon=lon)
    for k, v in s5p_const.items():
        df_full[k] = v
        
    unique_dates = df_full["timestamp"].dt.normalize().unique()
    s5p_dynamic_by_date = {}
    gee_success = False
    try:
        import ee
        for d in unique_dates:
            target_dt = pd.to_datetime(d)
            s5p_dynamic_by_date[d] = get_s5p_features(lat=lat, lon=lon, target_date=target_dt)
        gee_success = True
    except Exception as e:
        print(f"⚠️ Không thể tải S5P động theo ngày (sử dụng S5P hằng số): {e}")
        
    if gee_success:
        gas_keys = [
            "s5p_no2", "s5p_no2_cf", "s5p_so2", "s5p_so2_cf", "s5p_co", "s5p_co_cf", 
            "s5p_o3", "s5p_o3_cf", "s5p_aai", "s5p_aai_cf"
        ]
        for k in gas_keys:
            if k in df_full.columns:
                for d in unique_dates:
                    mask = df_full["timestamp"].dt.normalize() == d
                    if d in s5p_dynamic_by_date and k in s5p_dynamic_by_date[d]:
                        df_full.loc[mask, k] = s5p_dynamic_by_date[d][k]

    # Kỹ nghệ đặc trưng thời gian
    df_full = engineer_all_features(df_full, city=city)

    # 4. Sao lưu Ground Truth (24 tiếng tương lai thực tế - tức là ngày cuối cùng)
    df_ground_truth = df_full.tail(24).copy()
    actuals = {}
    for col in TARGET_COLUMNS:
        actuals[col] = df_ground_truth[col].tolist()

    # 5. Giả lập Inference (Đặt 24 tiếng tương lai = NaN để mô hình dự báo)
    df_inference = df_full.copy()
    for col in TARGET_COLUMNS:
        df_inference.loc[df_inference["time_idx"] >= 72, col] = np.nan
        
    # BƯỚC MỚI: Thời tiết tương lai của 24h cuối ĐÃ ĐƯỢC CHUYỂN SANG known_reals nên KHÔNG cần đặt bằng NaN nữa!
    # Điều này cho phép mô hình khai thác 100% thông tin dự báo thời tiết 24h để dự báo Ozone và bụi mịn chính xác cực độ.
    # unknown_weather = ["temp_c", "humidity_pct", "wind_speed_ms", "precipitation_mm", "pressure_hpa", "dewpoint_c"]
    # for col in unknown_weather:
    #     if col in df_inference.columns:
    #         df_inference.loc[df_inference["time_idx"] >= 72, col] = np.nan

    # Cực kỳ quan trọng: Đảm bảo các cờ phân loại có mặt để pandas không ép kiểu sang float64
    for col in ["is_foggy_risk", "is_smog_risk"]:
        if col in df_inference.columns:
            df_inference[col] = df_inference[col].fillna(0).astype(int)

    # 6. Khởi tạo Predictor và chạy dự báo
    print("⏳ Bước 3: Đang load mô hình TFT...")
    predictor = TFTPredictor(checkpoint_path=checkpoint_path, device="cpu")

    # Lấy mật độ giao thông từ TomTom API
    traffic_density = "normal"
    if apply_calibration:
        try:
            traffic_density = fetch_tomtom_traffic(lat, lon)
            print(f"🚦 Mật độ giao thông thực tế quét từ TomTom API: {traffic_density.upper()}")
        except Exception as e:
            print(f"⚠️ Không thể quét dữ liệu giao thông TomTom: {e}. Sử dụng mặc định: NORMAL")

    print(f"🧠 Bước 4: Đang chạy dự báo hồi cứu trên 24h thực tế vừa diễn ra ({'BẬT hiệu chỉnh' if apply_calibration else 'RAW, không hiệu chỉnh'}) | Mật độ giao thông: {traffic_density.upper()}...")
    result = predictor.predict(
        df_inference,
        dataset_params_path=params_path,
        apply_calibration=apply_calibration,
        traffic_density=traffic_density
    )

    # 7. So sánh & Tính toán sai số
    print("\n" + "═" * 80)
    print("📊 KẾT QUẢ LIVE VALIDATION - SO SÁNH THỰC TẾ VS DỰ BÁO CỦA AI")
    print(f"📍 Tọa độ test: ({lat}, {lon}) | Thành phố: {city.upper()}")
    print(f"🕒 Mốc xuất phát dự báo: {df_full['timestamp'].iloc[72]}")
    print("═" * 80)

    metrics_summary = []

    for target in TARGET_COLUMNS:
        name = TARGET_DISPLAY_NAMES[target]
        unit = TARGET_UNITS[target]

        actual_vals = np.array(actuals[target])
        pred_data = result["predictions"][name]
        p50_vals = np.array(pred_data["p50"])
        p10_vals = np.array(pred_data["p10"])
        p90_vals = np.array(pred_data["p90"])

        # Tính toán sai số
        mae = np.mean(np.abs(actual_vals - p50_vals))
        rmse = np.sqrt(np.mean((actual_vals - p50_vals) ** 2))
        
        # Đếm xem có bao nhiêu điểm thực tế nằm lọt vào dải tin cậy [P10, P90]
        within_bounds = np.sum((actual_vals >= p10_vals) & (actual_vals <= p90_vals))
        capture_pct = (within_bounds / len(actual_vals)) * 100

        metrics_summary.append({
            "Gas": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Capture %": round(capture_pct, 1),
            "Unit": unit
        })

        print(f"\n📈 Chỉ số {name} ({unit}):")
        print(f"   - Sai số tuyệt đối trung bình (MAE): {mae:.2f} {unit}")
        print(f"   - Sai số bình phương trung bình (RMSE): {rmse:.2f} {unit}")
        print(f"   - Tỷ lệ thực tế lọt vào dải an toàn P10-P90: {capture_pct:.1f}%")
        print("-" * 50)
        
        # Bảng so sánh 6 mốc giờ điển hình (cách nhau 4 tiếng)
        print(f"{'Mốc Giờ':<10}{'Thực tế':<12}{'Dự báo (P50)':<15}{'Dải P10-P90':<18}{'Trạng thái'}")
        for step in range(0, 24, 4):
            act = actual_vals[step]
            p50 = p50_vals[step]
            p10 = p10_vals[step]
            p90 = p90_vals[step]
            
            in_bound = "🟢 Trong dải" if (act >= p10 and act <= p90) else "🔴 Lệch dải"
            hour_str = f"+{step}h"
            print(f"{hour_str:<10}{act:<12.1f}{p50:<15.1f}[{p10:.1f}, {p90:.1f}]    {in_bound}")

    print("\n" + "═" * 80)
    print("📊 BẢNG TỔNG HỢP SAI SỐ LIVE TOÀN DIỆN")
    print("═" * 80)
    print(f"{'Chất khí':<10}{'MAE':<10}{'RMSE':<10}{'Lọt dải P10-P90 (%)':<25}{'Đơn vị'}")
    for m in metrics_summary:
        print(f"{m['Gas']:<10}{m['MAE']:<10}{m['RMSE']:<10}{m['Capture %']:<25}{m['Unit']}")
    print("═" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT Live Validation Tool")
    parser.add_argument("--lat", type=float, default=None, help="Vĩ độ")
    parser.add_argument("--lon", type=float, default=None, help="Kinh độ")
    parser.add_argument("--city", type=str, default="hanoi", choices=["hanoi", "hcmc"], help="Thành phố")
    parser.add_argument("--start", type=str, default="live", help="Mốc ngày bắt đầu chuỗi 96 tiếng ('live' hoặc 'YYYY-MM-DD')")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Đường dẫn file checkpoint")
    parser.add_argument("--params", type=str, default=DEFAULT_DATASET_PARAMS, help="Đường dẫn file dataset params")
    parser.add_argument("--calibration", action="store_true", help="Bật bộ hiệu chỉnh (mặc định tắt)")

    args = parser.parse_args()

    # Xác định tọa độ mặc định theo thành phố nếu người dùng không truyền vào
    lat = args.lat
    lon = args.lon
    if lat is None or lon is None:
        if args.city == "hcmc":
            lat = 10.7769
            lon = 106.7009
        else:
            lat = 21.0285
            lon = 105.8542

    run_live_test(
        lat=lat,
        lon=lon,
        city=args.city,
        start_date_str=args.start,
        checkpoint_path=args.checkpoint,
        params_path=args.params,
        apply_calibration=args.calibration,
    )
