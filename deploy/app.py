"""
deploy/app.py
=============
FastAPI Server — API dự báo chất lượng không khí 24 giờ.

Endpoints:
    POST /predict       - Nhận tọa độ (lat, lon) → Trả về dự báo 24h cho 6 chỉ số
    GET  /health        - Kiểm tra sức khỏe server
    GET  /              - Trang chủ hướng dẫn sử dụng

Chạy:
    cd deploy
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
from pathlib import Path

# Nếu gặp lỗi CUDA treo máy, bỏ comment dòng dưới để ép CPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Fix Unicode cho Windows Console (cp1252 không hỗ trợ emoji)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deploy.app")

# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🌤️ Air Quality Forecast API",
    description=(
        "API dự báo chất lượng không khí 24 giờ sử dụng mô hình "
        "Temporal Fusion Transformer (TFT) với 84 tính năng đầu vào.\n\n"
        "**Các chỉ số dự báo:** PM2.5, PM10, NO₂, SO₂, CO, O₃\n\n"
        "**Input:** Tọa độ GPS (latitude, longitude)\n\n"
        "**Output:** Dự báo theo giờ trong 24h tới với dải lượng tử P10/P50/P90"
    ),
    version="1.0.0",
)

# CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Vĩ độ (latitude)")
    lon: float = Field(..., ge=-180, le=180, description="Kinh độ (longitude)")
    city: Optional[str] = Field("hanoi", description="Thành phố: 'hanoi' hoặc 'hcmc'")
    apply_calibration: Optional[bool] = Field(True, description="Áp dụng hiệu chuẩn vật lý quang hóa O3 và dải tin cậy từng chất khí")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "lat": 21.0285,
                    "lon": 105.8542,
                    "city": "hanoi",
                    "apply_calibration": True
                }
            ]
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Global State ─────────────────────────────────────────────────────────────

# Đường dẫn mặc định tới checkpoint và dataset parameters
DEFAULT_CHECKPOINT = str(
    Path(__file__).parent / "models" / "tft-best-model-epoch=02-val_loss=132.2085.ckpt"
)
DEFAULT_DATASET_PARAMS = str(
    Path(__file__).parent / "models" / "training_dataset_params.pkl"
)

# Lazy-load predictor (chỉ load khi có request đầu tiên)
_predictor = None


def get_predictor():
    """Lazy-load TFT Predictor."""
    global _predictor
    if _predictor is None:
        checkpoint = os.environ.get("TFT_CHECKPOINT", DEFAULT_CHECKPOINT)
        logger.info(f"🔄 Lazy-loading TFT model từ: {checkpoint}")
        
        from deploy.predictor import TFTPredictor
        # Ép CPU cho inference (chỉ predict 1 batch duy nhất, CPU xử lý trong 0.1s, không cần GPU)
        _predictor = TFTPredictor(checkpoint_path=checkpoint, device="cpu")
    return _predictor


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Trang chủ với hướng dẫn sử dụng."""
    return {
        "message": "🌤️ Air Quality Forecast API",
        "version": "1.0.0",
        "usage": {
            "endpoint": "POST /predict",
            "body": {"lat": 21.0285, "lon": 105.8542, "city": "hanoi"},
            "description": "Gửi tọa độ GPS để nhận dự báo 24h cho 6 chỉ số ô nhiễm",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra sức khỏe server và trạng thái mô hình."""
    global _predictor
    return HealthResponse(
        status="healthy",
        model_loaded=_predictor is not None,
        device=_predictor.device if _predictor else "not loaded",
    )


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    🌍 Dự báo chất lượng không khí 24 giờ tới.
    
    Nhận tọa độ (lat, lon) và trả về dự báo cho 6 chỉ số:
    PM2.5, PM10, NO₂, SO₂, CO, O₃
    
    Mỗi chỉ số có 3 kịch bản:
    - **P10**: Kịch bản tốt nhất (10th percentile)
    - **P50**: Kịch bản trung bình (median)
    - **P90**: Kịch bản xấu nhất (90th percentile)
    """
    try:
        logger.info(f"📍 Nhận request: ({request.lat}, {request.lon}) | city={request.city}")

        # Bước 1: Xây dựng DataFrame inference
        from deploy.data_pipeline import build_inference_dataframe
        df = build_inference_dataframe(
            lat=request.lat,
            lon=request.lon,
            city=request.city or "hanoi",
        )

        # Bước 2: Chạy dự báo
        predictor = get_predictor()
        params_path = os.environ.get("TFT_DATASET_PARAMS", DEFAULT_DATASET_PARAMS)
        apply_calib = request.apply_calibration if request.apply_calibration is not None else True

        # Gọi TomTom Traffic Flow API nếu bật hiệu chỉnh
        traffic_density = "normal"
        if apply_calib:
            from deploy.data_pipeline import fetch_tomtom_traffic
            try:
                traffic_density = fetch_tomtom_traffic(request.lat, request.lon)
                logger.info(f"🚦 Mật độ giao thông quét từ TomTom API: {traffic_density.upper()}")
            except Exception as e:
                logger.warning(f"⚠️ Lỗi quét lưu lượng giao thông TomTom: {e}")

        result = predictor.predict(
            df,
            dataset_params_path=params_path,
            apply_calibration=apply_calib,
            traffic_density=traffic_density
        )

        # Bước 3: Thêm metadata
        result["location"] = {
            "lat": request.lat,
            "lon": request.lon,
            "city": request.city,
        }

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Mô hình chưa sẵn sàng: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Lỗi thu thập dữ liệu: {str(e)}")
    except Exception as e:
        logger.exception("Lỗi không xác định")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Mode (Chạy trực tiếp không cần FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

def cli_predict(lat: float, lon: float, city: str = "hanoi", apply_calibration: bool = True):
    """
    Chạy dự báo trực tiếp từ command line (không cần khởi động server).
    
    Ví dụ:
        python deploy/app.py --lat 21.0285 --lon 105.8542 --city hanoi
    """
    from deploy.data_pipeline import build_inference_dataframe
    from deploy.predictor import TFTPredictor, format_prediction_text

    checkpoint = os.environ.get("TFT_CHECKPOINT", DEFAULT_CHECKPOINT)
    params_path = os.environ.get("TFT_DATASET_PARAMS", DEFAULT_DATASET_PARAMS)

    print(f"\n🌍 Tọa độ: ({lat}, {lon}) | Thành phố: {city}")
    print(f"📦 Checkpoint: {checkpoint}")
    print(f"📖 Dataset Params: {params_path}\n")

    # Bước 1: Xây dựng DataFrame
    print("⏳ Đang thu thập dữ liệu thời tiết và AQI...")
    df = build_inference_dataframe(lat=lat, lon=lon, city=city)
    print(f"✅ DataFrame: {df.shape[0]} dòng × {df.shape[1]} cột\n")

    # Bước 2: Load model & predict
    print("⏳ Đang load mô hình TFT...")
    predictor = TFTPredictor(checkpoint_path=checkpoint)

    # Gọi TomTom Traffic Flow API
    traffic_density = "normal"
    if apply_calibration:
        from deploy.data_pipeline import fetch_tomtom_traffic
        try:
            traffic_density = fetch_tomtom_traffic(lat, lon)
            print(f"🚦 Mật độ giao thông quét từ TomTom API: {traffic_density.upper()}")
        except Exception as e:
            print(f"⚠️ Lỗi quét lưu lượng giao thông TomTom: {e}")

    print(f"⏳ Đang chạy dự báo 24h (Hiệu chuẩn={'BẬT' if apply_calibration else 'TẮT'}) | Mật độ giao thông: {traffic_density.upper()}...")
    result = predictor.predict(
        df,
        dataset_params_path=params_path,
        apply_calibration=apply_calibration,
        traffic_density=traffic_density
    )

    # Bước 3: In kết quả
    print(format_prediction_text(result))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Air Quality Forecast CLI")
    parser.add_argument("--lat", type=float, default=21.0285, help="Vĩ độ")
    parser.add_argument("--lon", type=float, default=105.8542, help="Kinh độ")
    parser.add_argument("--city", type=str, default="hanoi", help="Thành phố")
    parser.add_argument("--serve", action="store_true", help="Khởi động FastAPI server")
    parser.add_argument("--no-calibrate", action="store_true", help="Không áp dụng hiệu chuẩn sau dự báo")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        print("🚀 Khởi động FastAPI server...")
        uvicorn.run("deploy.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        cli_predict(lat=args.lat, lon=args.lon, city=args.city, apply_calibration=not args.no_calibrate)
