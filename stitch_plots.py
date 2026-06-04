"""
stitch_plots.py
===============
Kịch bản tiện ích dùng để ghép nhanh 6 ảnh biểu đồ quá trình học lẻ (đã được huấn luyện trước đó)
thành một khung hình lưới 2 hàng, 3 cột (2x3) để thuận tiện chèn vào báo cáo thực nghiệm.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def stitch_model_plots(model_type="xgboost"):
    model_dir = f"models/{model_type}"
    print(f"🔄 Đang tiến hành ghép 6 ảnh biểu đồ trong thư mục: {model_dir}...")
    
    # 6 chất khí theo đúng thứ tự báo cáo
    gases = ["pm25_obs", "pm10_obs", "no2_pseudo", "so2_pseudo", "co_pseudo", "o3_pseudo"]
    
    if not os.path.exists(model_dir):
        print(f"⚠️ Thư mục {model_dir} không tồn tại. Bỏ qua.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    found_count = 0
    for idx, gas in enumerate(gases):
        img_path = os.path.join(model_dir, f"learning_curve_{gas}.png")
        ax = axes[idx]
        
        if os.path.exists(img_path):
            # Đọc ảnh gốc
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off") # Tắt hiển thị viền/trục tọa độ của matplotlib cha
            found_count += 1
        else:
            ax.text(0.5, 0.5, f"Không tìm thấy:\nlearning_curve_{gas}.png", 
                    ha="center", va="center", color="red", fontsize=12)
            ax.axis("off")
            
    if found_count > 0:
        plt.tight_layout()
        output_path = os.path.join(model_dir, "learning_curves_combined.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Ghép ảnh thành công ({found_count}/6 ảnh)! Biểu đồ gộp đã được lưu tại: {output_path}")
    else:
        plt.close()
        print(f"❌ Không tìm thấy ảnh biểu đồ nào tại {model_dir} để thực hiện ghép.")

if __name__ == "__main__":
    # Tự động ghép cho cả hai mô hình (nếu tìm thấy các file ảnh)
    stitch_model_plots("lightgbm")
    stitch_model_plots("xgboost")
