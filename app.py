import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# Thêm thư mục src vào path để import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import preprocess_data
from models.cbla_dl import CBLA_DL_Model
from models.cbla_xgboost import CBLA_XGBoost
from evaluate import evaluate_metrics

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="CBLA Air Quality Prediction", page_icon="🌤️", layout="wide")
st.title("🌤️ Mô hình Lai CBLA Dự Báo Chất Lượng Không Khí (PM2.5)")
st.markdown("Đồ án Khám phá Dữ liệu - Xây dựng dựa trên bài báo **'Air quality prediction model based on deep learning hybrid framework'**.")

# --- SIDEBAR: ĐIỀU CHỈNH THAM SỐ ---
st.sidebar.header("⚙️ Tinh chỉnh Siêu Tham Số (Hyperparameters)")
epochs = st.sidebar.slider("Số Vòng Lặp Học Sâu (Epochs)", min_value=10, max_value=200, value=50, step=10)
lr = st.sidebar.number_input("Tốc độ học (Learning Rate)", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
xgb_max_depth = st.sidebar.slider("Độ sâu Cây XGBoost (Max Depth)", min_value=2, max_value=10, value=6, step=1)
xgb_estimators = st.sidebar.slider("Số lượng Cây XGBoost (n_estimators)", min_value=10, max_value=200, value=100, step=10)

start_btn = st.sidebar.button("🚀 Bắt đầu Huấn luyện Mô hình", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Gợi ý:** XGBoost sẽ hoạt động hiệu quả nhất khi kết hợp với mô hình Deep Learning đã hội tụ tốt (Epochs > 50).")

# --- HÀM HỖ TRỢ ---
def get_dl_predictions(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_attentions = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds, attn = model(X_batch)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())
            all_attentions.extend(attn.numpy())
    return np.array(all_preds), np.array(all_targets), np.array(all_attentions)

# --- MÀN HÌNH CHÍNH ---
data_path = "data/beijing_air_quality_2014.csv"
if os.path.exists(data_path):
    st.subheader("📊 Trích xuất Dữ liệu Gốc (Beijing Air Quality Dataset)")
    df_raw = pd.read_csv(data_path)
    st.dataframe(df_raw.head(10), use_container_width=True)
else:
    st.error("Không tìm thấy tệp dữ liệu. Vui lòng chạy lệnh sinh dữ liệu trước.")

if start_btn:
    with st.spinner("Đang tải và tiền xử lý dữ liệu (KNN Imputer & Sliding Windows)..."):
        # 1. Preprocess
        train_loader, test_loader, train_loader_unshuffled, scaler, X_train, y_train, X_test, y_test = preprocess_data(
            csv_path=data_path, time_steps=24
        )
    
    st.success(f"Đã xử lý xong: {len(X_train)} mẫu Train, {len(X_test)} mẫu Test.")
    
    # 2. Train DL
    st.subheader("🧠 Quá trình Huấn luyện Mô hình Học Sâu (CNN-BiLSTM-Attention)")
    dl_model = CBLA_DL_Model(input_nodes=11, time_steps=24)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(dl_model.parameters(), lr=lr)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    losses = []
    for epoch in range(epochs):
        dl_model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions, _ = dl_model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Update progress and chart
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Đang huấn luyện Epoch {epoch+1}/{epochs} | MAE Loss: {avg_loss:.4f}")
        
        # Vẽ biểu đồ loss thời gian thực
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(losses, color='blue', label='Train MAE')
            ax.set_title("Biểu đồ Suy giảm Sai số (Loss Curve)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MAE Loss")
            ax.legend()
            chart_placeholder.pyplot(fig)
            plt.close(fig)

    # 3. XGBoost
    with st.spinner("Đang huấn luyện mô hình XGBoost để tinh chỉnh dữ liệu Khí tượng..."):
        train_dl_preds, _, _ = get_dl_predictions(dl_model, train_loader_unshuffled)
        test_dl_preds, _, test_attentions = get_dl_predictions(dl_model, test_loader)
        
        met_feature_indices = [6, 7, 8, 9, 10]
        train_met_features = X_train[:, -1, met_feature_indices]
        test_met_features = X_test[:, -1, met_feature_indices]
        
        xgb_wrapper = CBLA_XGBoost(max_depth=xgb_max_depth, n_estimators=xgb_estimators)
        X_train_xgb = xgb_wrapper.prepare_features(train_dl_preds, train_met_features)
        X_test_xgb = xgb_wrapper.prepare_features(test_dl_preds, test_met_features)
        
        xgb_wrapper.fit(X_train_xgb, y_train)
        final_preds = xgb_wrapper.predict(X_test_xgb)
        
    st.success("Huấn luyện toàn bộ đường ống CBLA hoàn tất!")
    
    # 4. Hiển thị Kết quả
    st.markdown("---")
    st.subheader("🎯 Đánh giá Hiệu năng (Metrics)")
    
    dl_metrics = evaluate_metrics(y_test, test_dl_preds)
    cbla_metrics = evaluate_metrics(y_test, final_preds)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Mô hình Học Sâu Độc Lập**")
        st.metric("RMSE", f"{dl_metrics['RMSE']:.4f}")
        st.metric("MAE", f"{dl_metrics['MAE']:.4f}")
        st.metric("R² Score", f"{dl_metrics['R2']:.4f}")
        
    with col2:
        st.markdown("**Mô hình Lai CBLA (Kết hợp XGBoost)**")
        st.metric("RMSE", f"{cbla_metrics['RMSE']:.4f}", delta=f"{dl_metrics['RMSE'] - cbla_metrics['RMSE']:.4f}", delta_color="inverse")
        st.metric("MAE", f"{cbla_metrics['MAE']:.4f}", delta=f"{dl_metrics['MAE'] - cbla_metrics['MAE']:.4f}", delta_color="inverse")
        st.metric("R² Score", f"{cbla_metrics['R2']:.4f}", delta=f"{cbla_metrics['R2'] - dl_metrics['R2']:.4f}")

    # 5. Biểu đồ Dự đoán & Attention
    st.markdown("---")
    st.subheader("📈 Trực quan hóa Kết quả")
    
    tab1, tab2, tab3 = st.tabs(["Đường cong Dự báo", "Cơ chế Chú ý (Attention)", "Tầm quan trọng Đặc trưng (XGBoost)"])
    
    with tab1:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(y_test[:200], label='Thực tế (Actual PM2.5)', color='black', alpha=0.7)
        ax2.plot(final_preds[:200], label='Dự báo CBLA (Predicted)', color='red', alpha=0.7, linestyle='--')
        ax2.set_title("So sánh Thực tế và Dự báo (200 giờ đầu tiên của Test Set)")
        ax2.set_xlabel("Thời gian (Giờ)")
        ax2.set_ylabel("Nồng độ PM2.5 (Đã chuẩn hóa)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        
    with tab2:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.bar(range(24), test_attentions[0], color='orange')
        ax3.set_title("Trọng số Attention phân bổ trong 24 Giờ quá khứ (Mẫu ngẫu nhiên)")
        ax3.set_xlabel("Bước thời gian (0 = Xa nhất, 23 = Gần nhất)")
        ax3.set_ylabel("Trọng số Chú ý (Attention Weight)")
        ax3.grid(axis='y')
        st.pyplot(fig3)
        
    with tab3:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        # Các đặc trưng đưa vào XGBoost: [DL_Prediction, Temp, Humid, WindSpeed, WindDir, Weather]
        feature_names = ['Dự báo Học sâu', 'Nhiệt độ', 'Độ ẩm', 'Tốc độ gió', 'Hướng gió', 'Thời tiết']
        importances = xgb_wrapper.model.feature_importances_
        
        # Sắp xếp để vẽ biểu đồ ngang
        indices = np.argsort(importances)
        
        ax4.barh(range(len(indices)), importances[indices], color='teal')
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([feature_names[i] for i in indices])
        ax4.set_title("Mức độ ảnh hưởng của các Yếu tố (Feature Importance)")
        ax4.set_xlabel("Trọng số quan trọng (%)")
        st.pyplot(fig4)
