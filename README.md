# CBLA: Air Quality Prediction Model (Hybrid Deep Learning)

Đây là mã nguồn triển khai chính thức cho đồ án Khai phá dữ liệu, bám sát nội dung của bài báo khoa học: *"Air quality prediction model based on deep learning hybrid framework"*.

## Kiến trúc Mô hình (Model Architecture)
Dự án sử dụng cơ chế lai (Hybrid) kết hợp sức mạnh của 2 trường phái Học Máy:
1. **Khối Deep Learning (PyTorch):** Bao gồm mạng `1D-CNN` để trích xuất đặc trưng cục bộ, `BiLSTM` để học chuỗi thời gian 2 chiều (24 giờ), và `Bahdanau Attention Mechanism` để phân bổ trọng số động.
2. **Khối Tree-based (XGBoost):** Đóng vai trò lớp hiệu chỉnh thứ cấp, kết hợp kết quả của Deep Learning với 5 đặc trưng khí tượng để xử lý nhiễu phi tuyến tính.

## Cấu trúc Thư mục
```text
cbla_model/
├── data/                          # Tập dữ liệu đầu vào
│   └── beijing_air_quality_2014.csv
├── results/                       # Biểu đồ phân tích (Tự động sinh sau khi huấn luyện)
│   ├── training_loss.png          # Đường cong Loss giảm dần qua các Epochs
│   ├── prediction_curve.png       # So sánh Dự báo vs Thực tế (200 giờ)
│   ├── attention_weights.png      # Trọng số Attention 24 giờ
│   └── feature_importance.png     # Mức độ ảnh hưởng của các yếu tố (XGBoost)
├── saved_models/                  # Trọng số mô hình đã hội tụ
│   ├── cbla_dl_weights.pth        # Trọng số Deep Learning (PyTorch)
│   └── cbla_xgboost.json          # Cây quyết định XGBoost
├── src/                           # Lõi mã nguồn AI
│   ├── dataset.py                 # Tiền xử lý (KNN Imputer, MinMax, Sliding Window)
│   ├── evaluate.py                # Đánh giá chỉ số (RMSE, MAE, R²)
│   └── models/
│       ├── cbla_dl.py             # Kiến trúc CNN-BiLSTM-Attention (PyTorch)
│       └── cbla_xgboost.py        # Wrapper XGBoost
├── app.py                         # Giao diện Web tương tác (Streamlit)
├── train.py                       # Script huấn luyện qua dòng lệnh
└── README.md
```

## Hướng dẫn Cài đặt & Chạy
Yêu cầu hệ thống: `Python 3.8+`

### 1. Cài đặt thư viện
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn xgboost matplotlib streamlit
```

### 2. Chạy Giao diện Web (Khuyên dùng)
Giao diện trực quan cho phép tinh chỉnh siêu tham số và tự động vẽ biểu đồ phân tích.
```bash
streamlit run app.py
```

### 3. Chạy bằng Dòng lệnh (Terminal)
Dành cho kỹ sư muốn huấn luyện tự động với tham số mặc định của bài báo.
```bash
python3 train.py
```

## Kết quả Biểu đồ
Sau khi chạy `train.py` hoặc bấm "Bắt đầu Huấn luyện" trên Web, hệ thống tự động xuất 4 biểu đồ vào thư mục `results/`:

| Biểu đồ | Mô tả | Tham chiếu Bài báo |
| :--- | :--- | :--- |
| `training_loss.png` | Đường cong suy giảm sai số (MAE) qua 100 Epochs | Quá trình hội tụ mô hình |
| `prediction_curve.png` | So sánh đường Dự báo PM2.5 với Giá trị Thực tế | Hình 9, 10 |
| `attention_weights.png` | Phân bổ trọng số Attention trong 24 giờ quá khứ | Mục 3.3.3 |
| `feature_importance.png` | Mức độ ảnh hưởng các yếu tố khí tượng (XGBoost) | Hình 7 |

## 🔧 Siêu Tham Số Mặc Định (Theo Bảng 3 - Bài báo)
| Tham số | Giá trị |
| :--- | :--- |
| Time Steps (Cửa sổ trượt) | 24 giờ |
| CNN Filters | 16 |
| BiLSTM Hidden Units | 8 |
| Activation Function | Tanh |
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Epochs | 100 |
| XGBoost Max Depth | 6 |
| Optimizer | Adam |
| Loss Function | MAE (L1Loss) |

---
*Phát triển và tối ưu cho mục đích nghiên cứu khoa học & đồ án môn học - luongtd.tech@gmail.comcom.*
