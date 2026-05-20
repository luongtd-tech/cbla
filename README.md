# CBLA: Air Quality Prediction Model (Hybrid Deep Learning)

Đây là mã nguồn triển khai chính thức cho đồ án Khám phá Dữ liệu, bám sát kiến trúc Toán học của bài báo khoa học: *"Air quality prediction model based on deep learning hybrid framework"*.

## 🌟 Kiến trúc Mô hình (Model Architecture)
Dự án sử dụng cơ chế lai (Hybrid) kết hợp sức mạnh của 2 trường phái Học Máy:
1. **Khối Deep Learning (PyTorch):** Bao gồm mạng `1D-CNN` để trích xuất đặc trưng cục bộ, `BiLSTM` để học chuỗi thời gian 2 chiều (24 giờ), và `Bahdanau Attention Mechanism` để phân bổ trọng số động.
2. **Khối Tree-based (XGBoost):** Đóng vai trò lớp hiệu chỉnh thứ cấp, kết hợp kết quả của Deep Learning với 5 đặc trưng khí tượng để xử lý nhiễu phi tuyến tính.

## 📁 Cấu trúc Thư mục
- `data/`: Chứa tập dữ liệu Bắc Kinh 2014 (PM2.5 và Khí tượng).
- `src/`: Lõi mã nguồn AI (Tiền xử lý, Mạng Nơ-ron, Cây Quyết Định, Đánh giá Metrics).
- `saved_models/`: Trọng số mô hình sau khi hội tụ (`.pth` và `.json`).
- `results/`: Các biểu đồ phân tích trực quan hóa (Loss, Predictions, Attention Weights, Feature Importances).
- `app.py`: Giao diện Web tương tác (Streamlit).
- `train.py`: Mã kịch bản (Script) huấn luyện mô hình qua dòng lệnh.

## 🚀 Hướng dẫn Cài đặt & Chạy
Yêu cầu hệ thống: `Python 3.8+`

**1. Cài đặt thư viện**
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn xgboost matplotlib streamlit
```

**2. Chạy Giao diện Web (Khuyên dùng)**
Giao diện trực quan cho phép tinh chỉnh siêu tham số và tự động vẽ biểu đồ phân tích.
```bash
streamlit run app.py
```

**3. Chạy bằng Dòng lệnh (Terminal)**
Dành cho kỹ sư muốn huấn luyện tự động với tham số mặc định.
```bash
python3 train.py
```

---
*Phát triển và tối ưu cho mục đích Nghiên cứu Khoa học & Đồ án Đại học.*
