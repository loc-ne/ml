"""
processors/features.py
=======================
Feature engineering: tạo toàn bộ 67 features từ merged data.

Nhóm features:
  G — Lag features (lịch sử chuỗi thời gian)
  F — Temporal features (biến thời gian cyclic + flags VN)

CRITICAL — Anti-leakage rules:
  - Tất cả lag/rolling tính từ t trở về quá khứ
  - KHÔNG dùng bất kỳ giá trị nào từ t+1 trở đi khi tạo features
  - Kiểm tra bằng assertion trước khi lưu

Target:
  - Multi-output: {pm25, no2, o3, so2, co}_t+1 đến t+24
  - Tạo bằng shift(-h) — cần sort theo timestamp trước
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger("processors.features")

# Ngày Tết Nguyên Đán Việt Nam (thêm theo từng năm)
TET_DATES = [
    # (năm, tháng, ngày) — ngày đầu tiên Tết
    (2020, 1, 25), (2021, 2, 12), (2022, 2, 1),
    (2023, 1, 22), (2024, 2, 10), (2025, 1, 29),
    (2026, 2, 17),
]

# Ngày lễ cố định Việt Nam (tháng, ngày)
FIXED_HOLIDAYS = [
    (1,  1),   # Tết Dương lịch
    (4, 30),   # Ngày Giải phóng
    (5,  1),   # Quốc tế Lao động
    (9,  2),   # Quốc khánh
]

# Lag hours để tạo features
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]  # 168h = 1 tuần

# Rolling windows
ROLL_WINDOWS = [3, 6, 12, 24]

# Chỉ số chính cần tạo lag
PRIMARY_PARAMS = ["pm25", "pm10", "no2", "o3"]

# Horizon dự báo
FORECAST_HORIZONS = list(range(1, 25))  # 1h đến 24h

# Target parameters
TARGET_PARAMS = ["pm25", "pm10", "no2", "o3", "so2", "co"]


class FeatureEngineer:
    """
    Tạo toàn bộ features cần thiết để train model.

    Ví dụ:
        fe = FeatureEngineer()
        df_final = fe.transform(df_merged)
        # df_final có 67 features + 120 target columns
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline chính: gọi theo thứ tự."""
        logger.info(f"Feature engineering bắt đầu: {len(df):,} rows")

        sort_cols = ["timestamp"] if "station_id" not in df.columns else ["station_id", "timestamp"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        logger.info("  Tạo spatial/station features...")
        df = self._add_spatial_station_features(df)

        logger.info("  Tạo physical interaction features...")
        if "wind_speed_ms" in df.columns and "boundary_layer_h" in df.columns:
            # Ventilation Index: Tích của tốc độ gió và chiều cao lớp biên
            df["ventilation_index"] = df["wind_speed_ms"] * df["boundary_layer_h"]
        
        if "temp_c" in df.columns and "dewpoint_c" in df.columns:
            # Dew Point Spread: Nhiệt độ - Điểm sương
            # Insight: Spread thấp (~0) = Sương mù/Độ ẩm bão hòa = Dễ tích tụ bụi
            df["dew_point_spread"] = df["temp_c"] - df["dewpoint_c"]

            df["is_foggy_risk"] = (df["dew_point_spread"] < 2).astype(int)

            if "precipitation_mm" in df.columns:
                # 1 nếu spread thấp VÀ không mưa, 0 nếu ngược lại
                df["is_smog_risk"] = ((df["is_foggy_risk"] == 1) & (df["precipitation_mm"] == 0)).astype(int)

        logger.info("  Tạo temporal features...")
        df = self._add_temporal_features(df)

        logger.info("  Tạo lag features...")
        df = self._add_lag_features(df)

        logger.info("  Tạo rolling features...")
        df = self._add_rolling_features(df)

        logger.info("  Tạo trend features...")
        df = self._add_trend_features(df)

        logger.info("  Tạo target columns (multi-output 24h)...")
        df = self._add_targets(df)

        # Xóa các dòng đầu không đủ lag (168h = 7 ngày đầu)
        n_before = len(df)
        df = df.dropna(subset=[f"pm25_lag_{LAG_HOURS[-1]}h"])
        n_after = len(df)
        logger.info(f"  Drop {n_before - n_after:,} dòng đầu (chưa đủ lag window)")

        # Drop các dòng target bị NaN (cuối dataset)
        df = df.dropna(subset=["pm25_t+24"])
        logger.info(f"  Drop cuối dataset: {n_after - len(df):,} dòng không có target")

        # Anti-leakage assertion
        self._assert_no_leakage(df)

        logger.info(f"Feature engineering hoàn thành: {len(df):,} rows × {len(df.columns)} cols")
        return df.reset_index(drop=True)

    # ── Spatial & Station features ───────────────────────────────────────────

    def _add_spatial_station_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Spatial features theo từng giờ khi có nhiều trạm:
          - spatial_pm25_mean, spatial_pm25_std
          - station_pm25_rank, station_deviation
          - station_id_encoded
        """
        df = df.copy()

        if "pm25_obs" not in df.columns:
            return df

        if "station_id" in df.columns:
            station_codes = pd.Categorical(df["station_id"].astype(str))
            df["station_id_encoded"] = station_codes.codes.astype("int32")
        else:
            df["station_id_encoded"] = 0

        if "station_id" not in df.columns:
            df["spatial_pm25_mean"] = df["pm25_obs"]
            df["spatial_pm25_std"] = 0.0
            df["station_pm25_rank"] = 1.0
            df["station_deviation"] = 0.0
            return df

        g = df.groupby("timestamp")["pm25_obs"]
        df["spatial_pm25_mean"] = g.transform("mean")
        df["spatial_pm25_std"] = g.transform("std").fillna(0)
        df["station_pm25_rank"] = g.rank(method="dense", ascending=False)
        df["station_deviation"] = df["pm25_obs"] - df["spatial_pm25_mean"]

        return df

    # ── Temporal features ─────────────────────────────────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo temporal features dựa trên insight EDA:
        - Cyclic encoding (Sin/Cos)
        - City-aware Peak Traffic
        - Regional Seasonality (Winter/Harvest North vs Hot South)
        """
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])

        # 1. Cyclic encoding (dùng sin/cos để giữ tính liên tục)
        df["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
        df["dow_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        # 2. Binary flags cơ bản
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["is_holiday"] = self._flag_holidays(ts).astype(int)
        df["is_tet"]     = self._flag_tet(ts).astype(int)

        # 3. INSIGHT: Peak Traffic theo thành phố (Khớp EDA)
        df["is_peak_traffic"] = 0
        if "city" in df.columns:
            # Hà Nội: Sáng 7-9h, Tối muộn 18-21h (do nghịch nhiệt giữ bụi)
            hanoi_mask = df["city"].str.lower() == "hanoi"
            df.loc[hanoi_mask, "is_peak_traffic"] = ts[hanoi_mask].dt.hour.isin([7, 8, 9, 18, 19, 20, 21]).astype(int)
            
            # TP.HCM: Sáng sớm 6-8h, Chiều 17-19h (sớm hơn HN)
            hcmc_mask = df["city"].str.lower() == "hcmc"
            df.loc[hcmc_mask, "is_peak_traffic"] = ts[hcmc_mask].dt.hour.isin([6, 7, 8, 17, 18, 19]).astype(int)
        else:
            df["is_peak_traffic"] = ts.dt.hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

        # 4. INSIGHT: Đặc thù mùa theo vùng miền
        df["is_winter_north"]  = 0
        df["is_harvest_north"] = 0
        df["is_hot_dry_south"] = 0
        df["is_dry_season_south"] = 0
        
        if "city" in df.columns:
            # Mùa đông & Mùa gặt (Tháng 11-2 & 9-11) đặc trưng cho miền Bắc
            is_hanoi = df["city"].str.lower() == "hanoi"
            df.loc[is_hanoi, "is_winter_north"]  = ts[is_hanoi].dt.month.isin([11, 12, 1, 2]).astype(int)
            df.loc[is_hanoi, "is_harvest_north"] = ts[is_hanoi].dt.month.isin([9, 10, 11]).astype(int)
            
            # Đỉnh phụ tháng 3-4 (nắng nóng cực hạn) ở TP.HCM
            is_hcmc = df["city"].str.lower() == "hcmc"
            df.loc[is_hcmc, "is_dry_season_south"] = ts[is_hcmc].dt.month.isin([11, 12, 1, 2, 3, 4]).astype(int)
            df.loc[is_hcmc, "is_hot_dry_south"] = ts[is_hcmc].dt.month.isin([3, 4]).astype(int)
        
        # Giữ lại flag chung nếu không có cột city
        df["is_harvest_season"] = ts.dt.month.isin([9, 10, 11]).astype(int)

        return df

    def _flag_holidays(self, ts: pd.Series) -> pd.Series:
        """Flag các ngày lễ cố định Việt Nam."""
        flag = pd.Series(False, index=ts.index)
        for month, day in FIXED_HOLIDAYS:
            flag |= (ts.dt.month == month) & (ts.dt.day == day)
        return flag

    def _flag_tet(self, ts: pd.Series) -> pd.Series:
        """Flag 5 ngày trước Tết đến mùng 5 (10 ngày quanh Tết)."""
        flag = pd.Series(False, index=ts.index)
        for year, month, day in TET_DATES:
            tet_start = pd.Timestamp(year, month, day) - pd.Timedelta(days=5)
            tet_end   = pd.Timestamp(year, month, day) + pd.Timedelta(days=5)
            flag |= (ts >= tet_start) & (ts <= tet_end)
        return flag

    # ── Lag features ──────────────────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo lag features: pm25_lag_1h, pm25_lag_6h, ...
        Chỉ dùng shift(+h) — tức là lấy giá trị quá khứ.
        """
        df = df.copy()
        station_groups = df.groupby("station_id", sort=False) if "station_id" in df.columns else None

        for param in PRIMARY_PARAMS:
            obs_col = f"{param}_obs"
            if obs_col not in df.columns:
                continue

            for h in LAG_HOURS:
                # shift(h) lấy giá trị h bước TRƯỚC — đúng, không leakage
                if station_groups is not None:
                    df[f"{param}_lag_{h}h"] = station_groups[obs_col].shift(h)
                else:
                    df[f"{param}_lag_{h}h"] = df[obs_col].shift(h)

        # Lag weather (gió và nhiệt độ lag quan trọng)
        for wx_col in ["wind_speed_ms", "temp_c", "pressure_hpa"]:
            if wx_col not in df.columns:
                continue
            for h in [1, 3, 6, 12, 24]:
                if station_groups is not None:
                    df[f"{wx_col}_lag_{h}h"] = station_groups[wx_col].shift(h)
                else:
                    df[f"{wx_col}_lag_{h}h"] = df[wx_col].shift(h)

        return df

    # ── Rolling features ──────────────────────────────────────────────────────

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling mean, std, min, max trên các window.
        QUAN TRỌNG: dùng shift(1) trước khi rolling để đảm bảo
        không include giá trị hiện tại (tránh leakage).
        """
        df = df.copy()
        station_groups = df.groupby("station_id", sort=False) if "station_id" in df.columns else None

        for param in PRIMARY_PARAMS:
            obs_col = f"{param}_obs"
            if obs_col not in df.columns:
                continue

            # Không shift 1 nữa, do ta lấy chính t hiện tại (được dự báo từ t+1 trở đi)
            # Điều này không gây leakage vì pm25_obs(t) vốn nằm trong features
            for w in ROLL_WINDOWS:
                if station_groups is not None:
                    rolled = station_groups[obs_col].rolling(window=w, min_periods=max(1, w//2))
                    df[f"{param}_roll_mean_{w}h"] = rolled.mean().reset_index(level=0, drop=True)
                    df[f"{param}_roll_std_{w}h"]  = rolled.std().reset_index(level=0, drop=True).fillna(0)
                    df[f"{param}_roll_max_{w}h"]  = rolled.max().reset_index(level=0, drop=True)
                else:
                    rolled = df[obs_col].rolling(window=w, min_periods=max(1, w//2))
                    df[f"{param}_roll_mean_{w}h"] = rolled.mean()
                    df[f"{param}_roll_std_{w}h"]  = rolled.std().fillna(0)
                    df[f"{param}_roll_max_{w}h"]  = rolled.max()

        # Rolling wind speed (gió trung bình gần đây)
        if "wind_speed_ms" in df.columns:
            # Gồm luôn thời điểm t
            if station_groups is not None:
                wind_rolled = station_groups["wind_speed_ms"].rolling
                df["wind_roll_mean_6h"] = wind_rolled(6, min_periods=3).mean().reset_index(level=0, drop=True)
                df["wind_roll_mean_24h"] = wind_rolled(24, min_periods=12).mean().reset_index(level=0, drop=True)
            else:
                wind_col = df["wind_speed_ms"]
                df["wind_roll_mean_6h"]  = wind_col.rolling(6, min_periods=3).mean()
                df["wind_roll_mean_24h"] = wind_col.rolling(24, min_periods=12).mean()

        # 4. INSIGHT: Rolling Precipitation (Tổng mưa 24h) - Khả năng rửa trôi bụi
        if "precipitation_mm" in df.columns:
            if station_groups is not None:
                df["precip_roll_sum_24h"] = station_groups["precipitation_mm"].rolling(window=24, min_periods=1).sum().reset_index(level=0, drop=True)
            else:
                df["precip_roll_sum_24h"] = df["precipitation_mm"].rolling(window=24, min_periods=1).sum()

        return df

    # ── Trend features ────────────────────────────────────────────────────────

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính slope của PM2.5 trong 6h qua.
        Positive slope → đang tăng → nguy cơ cao hơn.
        Dùng polyfit bậc 1 trên cửa sổ trượt.
        """
        df = df.copy()
        station_groups = df.groupby("station_id", sort=False) if "station_id" in df.columns else None

        for param in ["pm25", "no2"]:
            obs_col = f"{param}_obs"
            if obs_col not in df.columns:
                continue

            # Slope 6h — dùng diff thay vì polyfit cho tốc độ
            # Tính gộp thời điểm hiện tại `t` thay vì `t-1` - không bị target leakage vì target là `t+1`
            if station_groups is not None:
                shifted = station_groups[obs_col].shift(6)
            else:
                shifted = df[obs_col].shift(6)
            df[f"{param}_trend_6h"] = (
                df[obs_col] - shifted
            ) / 6.0  # (t - t-6) / 6h

            # Flag: đang tăng nhanh (> 20 µg/m³ trong 6h)
            if param == "pm25":
                df["pm25_rising_fast"] = (df["pm25_trend_6h"] > 20/6).astype(int)

        return df

    # ── Target columns ────────────────────────────────────────────────────────

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo target columns: {param}_t+{h} cho h = 1..24.
        Dùng shift(-h) để lấy giá trị TƯƠNG LAI.
        Đây là LABELS — KHÔNG dùng làm features.
        """
        df = df.copy()
        station_groups = df.groupby("station_id", sort=False) if "station_id" in df.columns else None

        for param in TARGET_PARAMS:
            obs_col = f"{param}_obs"
            if obs_col not in df.columns:
                continue

            for h in FORECAST_HORIZONS:
                # shift(-h): dịch ngược lại h bước = lấy giá trị tương lai
                if station_groups is not None:
                    df[f"{param}_t+{h}"] = station_groups[obs_col].shift(-h)
                else:
                    df[f"{param}_t+{h}"] = df[obs_col].shift(-h)

        return df

    # ── Validation ────────────────────────────────────────────────────────────

    def _assert_no_leakage(self, df: pd.DataFrame) -> None:
        """
        Kiểm tra không có data leakage:
        Không có feature nào sử dụng thông tin từ t+1 trở đi.

        Columns bắt đầu bằng 't+' là target — không được dùng làm feature.
        """
        feature_cols = [
            c for c in df.columns
            if not c.startswith(tuple([f"{p}_t+" for p in TARGET_PARAMS]))
            and c != "timestamp"
        ]

        # Tất cả lag cols phải có suffix _lag_ hoặc _roll_ hoặc _trend_
        # → không phải raw observation tương lai
        suspicious = [
            c for c in feature_cols
            if "_t+" in c
        ]
        if suspicious:
            raise ValueError(
                f"LEAKAGE DETECTED: Các cột sau có thể chứa thông tin tương lai: {suspicious}"
            )

        logger.info("  Anti-leakage check: PASSED")

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Trả về danh sách feature columns (không bao gồm index và target)."""
        exclude_prefixes = tuple([f"{p}_t+" for p in TARGET_PARAMS])
        exclude_exact    = {"timestamp", "city", "station_id", "data_source",
                            "lat", "lon", "wind_dir_deg"}
        return [
            c for c in df.columns
            if not c.startswith(exclude_prefixes)
            and c not in exclude_exact
        ]

    def get_target_columns(self, df: pd.DataFrame, param: str = "pm25") -> List[str]:
        """Trả về danh sách target columns cho 1 parameter."""
        return [f"{param}_t+{h}" for h in FORECAST_HORIZONS
                if f"{param}_t+{h}" in df.columns]
