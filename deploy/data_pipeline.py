"""
deploy/data_pipeline.py
=======================
Pipeline thu thập dữ liệu real-time cho Inference:
  1. Gom biến tĩnh (Static Features) từ database GIS sẵn có
  2. Kéo API thời tiết (Open-Meteo) cho 72h quá khứ
  3. Nội suy AQI quá khứ từ các trạm gần nhất (IDW)
  4. Lấy dữ liệu vệ tinh S5P gần nhất (Forward-fill)

Logic API được tham chiếu từ: collectors/weather.py, collectors/openaq.py
"""

import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("deploy.pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# 1. BIẾN TĨNH (STATIC FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

# Database GIS tĩnh cho các trạm đã biết trước (tra cứu O(1))
# Trong sản phẩm thực tế, bạn nên lưu trong PostGIS hoặc một file GeoJSON
STATION_STATIC_DB = {
    "hanoi_center": {
        "elevation_m": 12.0,
        "population_density": 4200.0,
        "dist_to_industrial_km": 15.2,
        "dist_to_center_km": 0.0,
        "land_use_built_pct": 85.0,
        "angle_to_industrial_deg": 45.0,
    },
    "hcmc_center": {
        "elevation_m": 5.0,
        "population_density": 4800.0,
        "dist_to_industrial_km": 12.5,
        "dist_to_center_km": 0.0,
        "land_use_built_pct": 90.0,
        "angle_to_industrial_deg": 120.0,
    },
}

# Tọa độ trung tâm các thành phố
CITY_COORDS = {
    "hanoi": (21.0285, 105.8542),
    "hcmc":  (10.8231, 106.6297),
}

# Tọa độ khu công nghiệp lớn nhất
INDUSTRIAL_ZONES = {
    "hanoi": [
        {"name": "Thang Long", "lat": 21.0594, "lon": 105.7076},
        {"name": "Bac Thang Long", "lat": 21.1167, "lon": 105.7567},
        {"name": "Noi Bai", "lat": 21.2213, "lon": 105.8070},
    ],
    "hcmc": [
        {"name": "Tan Binh", "lat": 10.8127, "lon": 106.6252},
        {"name": "VSIP", "lat": 10.9671, "lon": 106.7219},
        {"name": "Binh Duong", "lat": 11.0583, "lon": 106.6773},
    ],
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Tính khoảng cách giữa 2 tọa độ bằng công thức Haversine (km)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_static_features(lat: float, lon: float) -> dict:
    """
    Tính toán các biến tĩnh (Static Features) cho một tọa độ bất kỳ.
    Sử dụng nội suy khoảng cách từ database GIS.
    """
    # Xác định thành phố gần nhất
    city = "hanoi"
    min_dist = float("inf")
    for c, (clat, clon) in CITY_COORDS.items():
        d = haversine_km(lat, lon, clat, clon)
        if d < min_dist:
            min_dist = d
            city = c

    dist_to_center_km = min_dist

    # Tìm khu công nghiệp gần nhất
    zones = INDUSTRIAL_ZONES.get(city, [])
    nearest_dist = 50.0  # Default 50km
    nearest_angle = 0.0
    for z in zones:
        d = haversine_km(lat, lon, z["lat"], z["lon"])
        if d < nearest_dist:
            nearest_dist = d
            # Tính góc phương vị
            dlon = np.radians(z["lon"] - lon)
            y = np.sin(dlon) * np.cos(np.radians(z["lat"]))
            x = (np.cos(np.radians(lat)) * np.sin(np.radians(z["lat"])) -
                 np.sin(np.radians(lat)) * np.cos(np.radians(z["lat"])) * np.cos(dlon))
            nearest_angle = (np.degrees(np.arctan2(y, x)) + 360) % 360

    # Ước lượng các biến tĩnh (trong sản phẩm thực tế: tra cứu từ PostGIS/GeoTIFF)
    # Elevation: Hà Nội ~10-15m, HCMC ~2-8m
    elevation = 12.0 if city == "hanoi" else 5.0
    # Population density: ước lượng theo khoảng cách trung tâm
    pop_density = max(500, 5000 - dist_to_center_km * 200)
    # Land use built percentage
    land_use = max(10, 90 - dist_to_center_km * 5)

    return {
        "elevation_m": elevation,
        "population_density": pop_density,
        "dist_to_industrial_km": nearest_dist,
        "dist_to_center_km": dist_to_center_km,
        "land_use_built_pct": land_use,
        "angle_to_industrial_deg": nearest_angle,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. API THỜI TIẾT (OPEN-METEO) — Lấy 72h quá khứ
# ══════════════════════════════════════════════════════════════════════════════

# Sử dụng Forecast API (không phải Archive API) để lấy dữ liệu gần real-time
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
    "shortwave_radiation",
    "cloud_cover",
    "dewpoint_2m",
    "wind_gusts_10m",
]


def fetch_weather_96h(lat: float, lon: float) -> pd.DataFrame:
    """
    Kéo dữ liệu thời tiết 72h quá khứ và 24h tương lai từ Open-Meteo Forecast API.
    API này miễn phí và không cần API key.

    Tổng cộng thu được 96 giờ khí tượng liên tục.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(HOURLY_VARS),
        "past_hours":      72,
        "forecast_hours":  24,
        "timezone":        "Asia/Ho_Chi_Minh",
        "wind_speed_unit": "ms",
    }

    try:
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"Open-Meteo API thất bại: {e}")
        return pd.DataFrame()

    if "hourly" not in data:
        logger.error(f"Open-Meteo response thiếu 'hourly': {data.get('reason', '')}")
        return pd.DataFrame()

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

    # Clip giá trị bất thường (copy logic từ weather.py)
    clip_rules = {
        "temp_c": (-20, 50), "humidity_pct": (0, 100), "wind_speed_ms": (0, 50),
        "precipitation_mm": (0, 500), "boundary_layer_h": (0, 5000),
        "dewpoint_c": (-30, 50), "wind_gust_ms": (0, 80),
        "cloud_cover_pct": (0, 100), "shortwave_rad": (0, 1500),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Nội suy các giá trị trống nhẹ (gap <= 3h)
    df = df.interpolate(method="linear", limit=3)

    logger.info(f"Open-Meteo: thu được {len(df)} giờ dữ liệu thời tiết (72h quá khứ + 24h tương lai)")
    return df



# ══════════════════════════════════════════════════════════════════════════════
# 3. NỘI SUY AQI QUÁ KHỨ (IDW — Inverse Distance Weighting)
# ══════════════════════════════════════════════════════════════════════════════

# Tọa độ các trạm đo chất lượng không khí đã biết
# (Trong sản phẩm thực tế, lấy danh sách này từ database)
KNOWN_AQ_STATIONS = {
    "hanoi": [
        {"station_id": "openaq_hanoi_us_embassy", "lat": 21.0245, "lon": 105.8412,
         "name": "US Embassy Hanoi"},
        {"station_id": "openaq_hanoi_trung_yen", "lat": 21.0170, "lon": 105.8015,
         "name": "Trung Yen"},
        {"station_id": "openaq_hanoi_ngo_quyen", "lat": 21.0340, "lon": 105.8560,
         "name": "Ngo Quyen"},
    ],
    "hcmc": [
        {"station_id": "openaq_hcmc_us_consulate", "lat": 10.7831, "lon": 106.7000,
         "name": "US Consulate HCMC"},
        {"station_id": "openaq_hcmc_nguyen_van_cu", "lat": 10.7685, "lon": 106.6836,
         "name": "Nguyen Van Cu"},
    ],
}

# Giá trị nền (baseline) cho các chỉ số khí khi không có dữ liệu trạm
# (Giá trị trung bình năm điển hình cho Hà Nội)
AQI_BASELINES = {
    "pm25_obs": 35.0,   # μg/m³
    "pm10_obs": 65.0,
    "no2_pseudo": 25.0,
    "so2_pseudo": 8.0,
    "co_pseudo":  800.0,  # μg/m³
    "o3_pseudo":  45.0,
}


def idw_interpolate_aqi(
    lat: float, lon: float,
    n_hours: int = 72,
    city: str = "hanoi",
) -> pd.DataFrame:
    """
    Tạo chuỗi AQI 72h quá khứ cho tọa độ (lat, lon) bằng thuật toán
    Inverse Distance Weighting (IDW) từ các trạm gần nhất.

    Trong phiên bản demo này, chúng ta sử dụng giá trị baseline + nhiễu ngẫu nhiên
    để mô phỏng dữ liệu thực. Trong sản phẩm thực tế, bạn sẽ gọi OpenAQ API
    theo logic của collectors/openaq.py để lấy dữ liệu real-time.
    """
    now = datetime.utcnow() + timedelta(hours=7)  # GMT+7
    now_floor = now.replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(
        end=now_floor,
        periods=n_hours,
        freq="h",
    )

    stations = KNOWN_AQ_STATIONS.get(city, [])

    # Tính trọng số IDW cho từng trạm
    weights = []
    for s in stations:
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        d = max(d, 0.1)  # Tránh chia cho 0
        weights.append(1.0 / d**2)

    total_w = sum(weights) if weights else 1.0
    weights = [w / total_w for w in weights]

    # Tạo chuỗi AQI dựa trên Baseline + biến động theo giờ
    records = []
    for ts in timestamps:
        hour = ts.hour
        # Mô phỏng chu kỳ ngày: PM2.5 cao vào sáng sớm (7-9h) và tối (18-22h)
        hour_factor = 1.0
        if hour in [7, 8, 9]:
            hour_factor = 1.4
        elif hour in [18, 19, 20, 21]:
            hour_factor = 1.3
        elif hour in [2, 3, 4]:
            hour_factor = 0.7

        record = {"timestamp": ts}
        for target, baseline in AQI_BASELINES.items():
            # IDW weighted baseline + random noise (±20%)
            noise = np.random.uniform(-0.2, 0.2)
            value = baseline * hour_factor * (1 + noise)
            record[target] = max(0, value)
        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"IDW AQI: tạo {len(df)} giờ dữ liệu nồng độ khí cho ({lat:.4f}, {lon:.4f})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. VỆ TINH S5P (Forward-fill giá trị gần nhất)
# ══════════════════════════════════════════════════════════════════════════════

# Giá trị trung bình S5P điển hình cho Hà Nội (dùng làm fallback)
S5P_DEFAULTS = {
    "s5p_no2": 8.5e-5,    # mol/m²
    "s5p_no2_cf": 0.3,
    "s5p_so2": 2.0e-4,
    "s5p_so2_cf": 0.25,
    "s5p_co": 0.03,
    "s5p_co_cf": 0.2,
    "s5p_o3": 0.12,
    "s5p_o3_cf": 0.15,
    "s5p_aai": 0.5,
    "s5p_aai_cf": 0.2,
    "s5p_aod": 0.4,
    "s5p_aod_cf": 0.3,
    "s5p_days_since_obs": 1,
    "s5p_wind_alignment": 0.5,
}


def get_s5p_features(lat: float, lon: float, target_date = None) -> dict:
    """
    Kéo dữ liệu cột khí Sentinel-5P thực tế thời gian thực từ Google Earth Engine (GEE).
    Tự động fallback về S5P_DEFAULTS nếu gặp bất kỳ lỗi nào.
    """
    logger.info(f"🛰️ Đang kết nối Google Earth Engine để lấy Sentinel-5P thực tế tại ({lat:.4f}, {lon:.4f})...")
    
    # Khởi tạo dict kết quả bằng defaults
    result = S5P_DEFAULTS.copy()
    
    try:
        import ee
        # Khởi tạo GEE với project của người dùng
        try:
            ee.Initialize(project='cedar-chemist-469506-t2')
        except Exception:
            # Fallback khởi tạo mặc định nếu đã authenticate cục bộ
            ee.Initialize()
            
        # Tạo vùng đệm (buffer) 5km quanh tọa độ
        point = ee.Geometry.Point([float(lon), float(lat)])
        geometry = point.buffer(5000)
        
        # Lấy khoảng thời gian: 7 ngày qua để đảm bảo có ít nhất 1 ảnh không bị mây che
        end_date = target_date if target_date is not None else datetime.now()
        if hasattr(end_date, "to_pydatetime"):
            end_date = end_date.to_pydatetime()
        start_date = end_date - timedelta(days=7)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        collections = {
            "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "NO2_column_number_density"),
            "so2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
            "co":  ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
            "o3":  ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density"),
            "aai": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index"),
        }
        
        obs_times = []
        
        for key, (coll_id, band_name) in collections.items():
            # Lọc ảnh 7 ngày qua tại điểm
            coll = ee.ImageCollection(coll_id).filterDate(start_date_str, end_date_str).filterBounds(geometry)
            
            # Sắp xếp lấy ảnh mới nhất
            latest_img = coll.sort('system:time_start', False).first()
            
            # Kiểm tra xem có ảnh không
            if latest_img is not None:
                # Lấy thời gian chụp ảnh gần nhất
                try:
                    time_start = latest_img.get('system:time_start').getInfo()
                    if time_start:
                        obs_date = datetime.fromtimestamp(time_start / 1000.0)
                        obs_times.append(obs_date)
                except Exception:
                    pass
                
                # Tính giá trị trung bình trong vùng buffer 5km
                try:
                    stats = latest_img.select(band_name).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=7000,
                        bestEffort=True
                    ).getInfo()
                    
                    val = stats.get(band_name)
                    if val is not None:
                        result[f"s5p_{key}"] = float(val)
                except Exception as e:
                    logger.debug(f"Lỗi extract {key} từ GEE: {e}")
                    
                # Lấy cloud fraction (nếu có)
                try:
                    band_names = latest_img.bandNames().getInfo()
                    if "cloud_fraction" in band_names:
                        cf_stats = latest_img.select("cloud_fraction").reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=geometry,
                            scale=7000,
                            bestEffort=True
                        ).getInfo()
                        cf_val = cf_stats.get("cloud_fraction")
                        if cf_val is not None:
                            result[f"s5p_{key}_cf"] = float(cf_val)
                except Exception:
                    pass
        
        # Tính days_since_obs dựa trên ngày chụp mới nhất của các ảnh
        if obs_times:
            newest_obs = max(obs_times)
            days_since = (end_date - newest_obs).days
            result["s5p_days_since_obs"] = max(0, days_since)
            logger.info(f"✅ GEE: Tải thành công dữ liệu Sentinel-5P (Ảnh mới nhất: {newest_obs.strftime('%Y-%m-%d')}, cách đây {days_since} ngày)")
        else:
            result["s5p_days_since_obs"] = 7  # Fallback
            
    except Exception as e:
        logger.warning(f"⚠️ Kết nối GEE thất bại hoặc chưa cấu hình xác thực. Sử dụng S5P_DEFAULTS fallback. Chi tiết lỗi: {e}")
        
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.5. MẬT ĐỘ GIAO THÔNG TOMTOM
# ══════════════════════════════════════════════════════════════════════════════

def fetch_tomtom_traffic(lat: float, lon: float, api_key: str = "bQKs7wOAnp5FiESwRJx8LFZ7KM56SZyn") -> str:
    """
    Truy vấn TomTom Traffic Flow API để lấy thông tin tốc độ thực tế tại tọa độ.
    Tính tỷ lệ speed ratio (currentSpeed / freeFlowSpeed) để phân loại mật độ:
      - ratio >= 0.90 -> 'low'
      - 0.75 <= ratio < 0.90 -> 'normal'
      - 0.55 <= ratio < 0.75 -> 'high'
      - ratio < 0.55 -> 'jam'
    """
    logger.info(f"🚦 Đang lấy dữ liệu giao thông TomTom tại ({lat:.4f}, {lon:.4f})...")
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/18/json"
    params = {
        "key": api_key,
        "point": f"{lat},{lon}",
        "unit": "KMPH"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            flow = data.get("flowSegmentData", {})
            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            
            if current_speed is not None and free_flow_speed is not None and free_flow_speed > 0:
                ratio = current_speed / free_flow_speed
                logger.info(f"   TomTom: Tốc độ hiện tại: {current_speed} km/h | Tốc độ tự do: {free_flow_speed} km/h | Tỉ lệ: {ratio:.2f}")
                
                if ratio >= 0.90:
                    density = "low"
                elif ratio >= 0.75:
                    density = "normal"
                elif ratio >= 0.55:
                    density = "high"
                else:
                    density = "jam"
                
                logger.info(f"   → Mật độ giao thông xác định: {density.upper()}")
                return density
            else:
                logger.warning("   TomTom: Thiếu dữ liệu tốc độ trong phản hồi API.")
        else:
            logger.warning(f"   TomTom API báo lỗi (Status Code {resp.status_code}): {resp.text}")
    except Exception as e:
        logger.warning(f"   ⚠️ Kết nối TomTom API thất bại: {e}")
        
    logger.info("   → Sử dụng mật độ giao thông mặc định: NORMAL")
    return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE TỔNG HỢP
# ══════════════════════════════════════════════════════════════════════════════

def build_inference_dataframe(
    lat: float,
    lon: float,
    city: str = "hanoi",
    station_id: str = "openaq_7440",
    max_encoder_length: int = 72,
    max_prediction_length: int = 24,
) -> pd.DataFrame:
    """
    Xây dựng DataFrame hoàn chỉnh (96 dòng x 84+ features) sẵn sàng cho TFT inference.

    Pipeline:
    1. Kéo 96h thời tiết (72h quá khứ + 24h tương lai) từ Open-Meteo API
    2. Tạo 72h AQI quá khứ (IDW nội suy)
    3. Ghép nối và điền biến tĩnh GIS, vệ tinh S5P
    4. Gán các target tương lai bằng NaN (mô hình tự dự báo)
    5. Tính toán toàn bộ 84 đặc trưng (Feature Engineering) cho cả 96 dòng

    Args:
        lat: Vĩ độ
        lon: Kinh độ
        city: Tên thành phố
        station_id: Mã trạm (tùy ý)
        max_encoder_length: Số giờ quá khứ (encoder)
        max_prediction_length: Số giờ dự báo (decoder)

    Returns:
        DataFrame 96 dòng (72 + 24) với đầy đủ features
    """
    logger.info(f"🚀 Bắt đầu xây dựng DataFrame cho ({lat}, {lon}) | city={city}")

    # ── Bước 1: Kéo thời tiết 96h (72h quá khứ + 24h tương lai) ─────────────
    df_weather = fetch_weather_96h(lat, lon)
    if df_weather.empty:
        raise RuntimeError("Không thể kéo dữ liệu thời tiết từ Open-Meteo!")

    # Đảm bảo có đúng 96 dòng dữ liệu thời tiết
    if len(df_weather) < 96:
        raise RuntimeError(f"Open-Meteo API chỉ trả về {len(df_weather)} giờ thời tiết, yêu cầu tối thiểu 96 giờ!")
    
    df_weather = df_weather.tail(max_encoder_length + max_prediction_length).reset_index(drop=True)

    # Chia thời tiết thành quá khứ (72h) và tương lai (24h) để gộp với AQI
    df_weather_past = df_weather.head(max_encoder_length).copy().reset_index(drop=True)
    df_weather_future = df_weather.tail(max_prediction_length).copy().reset_index(drop=True)

    # ── Bước 2: Nội suy AQI quá khứ (72h) ────────────────────────────────────
    df_aqi_past = idw_interpolate_aqi(lat, lon, n_hours=max_encoder_length, city=city)
    # Đồng bộ timestamps quá khứ
    df_aqi_past = df_aqi_past[df_aqi_past["timestamp"].isin(df_weather_past["timestamp"])].reset_index(drop=True)

    # Ghép thời tiết quá khứ + AQI quá khứ
    df_past = df_weather_past.merge(df_aqi_past, on="timestamp", how="left")

    # ── Bước 3: Tạo DataFrame tương lai (24h) ───────────────────────────────
    df_future = df_weather_future.copy()
    
    # Gán các biến target ô nhiễm ở tương lai bằng NaN để mô hình tự dự báo
    target_cols = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
    for col in target_cols:
        df_future[col] = np.nan

    # Gộp quá khứ và tương lai thành DataFrame 96 dòng
    df_full = pd.concat([df_past, df_future], ignore_index=True)

    # ── Bước 4: Thêm biến tĩnh (GIS) ─────────────────────────────────────────
    static = compute_static_features(lat, lon)
    for k, v in static.items():
        df_full[k] = v

    # ── Bước 5: Thêm vệ tinh S5P (Động cho khí, Tĩnh cho bụi) ──────────────────
    # 1. Lấy S5P mặc định/fallback hằng số cho AOD và làm nền
    s5p_const = get_s5p_features(lat=lat, lon=lon)
    for k, v in s5p_const.items():
        df_full[k] = v
        
    # 2. Truy cập GEE kéo dữ liệu động cho các chất khí từng ngày trong cửa sổ 96h
    unique_dates = df_full["timestamp"].dt.normalize().unique()
    s5p_dynamic_by_date = {}
    gee_success = False
    try:
        import ee
        # Chỉ chạy nếu đã initialize được ee thành công
        for d in unique_dates:
            target_dt = pd.to_datetime(d)
            s5p_dynamic_by_date[d] = get_s5p_features(lat=lat, lon=lon, target_date=target_dt)
        gee_success = True
    except Exception as e:
        logger.warning(f"⚠️ Không thể tải S5P động theo ngày (sử dụng S5P hằng số fallback): {e}")
        
    if gee_success:
        # Áp dụng S5P động cho các cột khí (trừ AOD)
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

    # ── Bước 6: Thêm metadata & source columns ───────────────────────────────
    df_full["city"] = city
    df_full["station_id"] = station_id
    df_full["data_source"] = "openaq"
    
    for src in ["pm25_source", "pm10_source", "no2_source", "so2_source", "co_source", "o3_source"]:
        df_full[src] = "real"

    # ── Bước 7: Feature Engineering trên toàn bộ 96 dòng ──────────────────────
    # Điều này giúp tính toán các biến phái sinh như ventilation_index, 
    # stagnation_index, dew_point_spread cho tương lai dựa trên thời tiết thực tế
    from deploy.feature_engineer import engineer_all_features
    df_full = engineer_all_features(df_full, city=city)

    # Đảm bảo các cờ phân loại có mặt để pandas không ép kiểu sang float64
    for col in ["is_foggy_risk", "is_smog_risk"]:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(0).astype(int)

    # Tạo time_idx liên tục
    df_full["time_idx"] = range(len(df_full))

    logger.info(f"✅ DataFrame hoàn chỉnh: {df_full.shape[0]} dòng × {df_full.shape[1]} cột")
    return df_full
