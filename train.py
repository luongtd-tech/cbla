import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from src.dataset import preprocess_data
from src.models.cbla_dl import CBLA_DL_Model
from src.models.cbla_xgboost import CBLA_XGBoost
from src.evaluate import evaluate_metrics, print_evaluation

def train_dl_model(model, train_loader, epochs=100, lr=0.001):
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    print(f"{'Epoch':<10} | {'MAE Loss':<15}")
    print("-" * 25)
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions, _ = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<10} | {avg_loss:<15.4f}")
            
    return losses

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

def plot_results(losses, y_true, y_pred, attn_weights, xgb_model):
    os.makedirs("results", exist_ok=True)
    
    # 1. Plot Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training MAE Loss', color='blue', linewidth=2)
    plt.title('Deep Learning Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_loss.png')
    plt.close()

    # 2. Plot True vs Predicted (First 200 samples)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:200], label='Actual PM2.5', color='black', alpha=0.7)
    plt.plot(y_pred[:200], label='Predicted PM2.5 (CBLA)', color='red', alpha=0.7, linestyle='--')
    plt.title('PM2.5 Prediction vs Actual (First 200 hours)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Normalized PM2.5')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/prediction_curve.png')
    plt.close()
    
    # 3. Plot Attention Weights for a single sample
    plt.figure(figsize=(10, 4))
    plt.bar(range(24), attn_weights[0], color='orange')
    plt.title('Attention Weights across 24-hour Time Steps')
    plt.xlabel('Time Step (0 is oldest, 23 is most recent)')
    plt.ylabel('Attention Weight')
    plt.grid(axis='y')
    plt.savefig('results/attention_weights.png')
    plt.close()

    # 4. Plot XGBoost Feature Importance (Hình 7 trong bài báo)
    plt.figure(figsize=(8, 5))
    feature_names = ['DL Prediction', 'Temp', 'Humid', 'WindSpeed', 'WindDir', 'Weather']
    importances = xgb_model.model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance Score')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

    print("\n[INFO] Da luu 4 bieu do phan tich tai thu muc 'results/'.")

def main():
    print("="*50)
    print(" BAT DAU HUAN LUYEN MO HINH LAI CBLA (HOC SAU + XGBOOST)")
    print("="*50)
    
    print("\n--- 1. Tien xu ly du lieu (KNN Imputer & Sliding Windows) ---")
    train_loader, test_loader, train_loader_unshuffled, scaler, X_train, y_train, X_test, y_test = preprocess_data(
        csv_path="data/beijing_air_quality_2014.csv", time_steps=24
    )
    print(f"So mau huan luyen (Train): {len(X_train)} | So mau kiem thu (Test): {len(X_test)}")
    
    print("\n--- 2. Huan luyen khoi Deep Learning (1D-CNN + BiLSTM + Attention) ---")
    dl_model = CBLA_DL_Model(input_nodes=11, time_steps=24)
    losses = train_dl_model(dl_model, train_loader, epochs=100, lr=0.001)
    
    print("\n--- 3. Trich xuat dac trung va du bao so bo ---")
    train_dl_preds, _, _ = get_dl_predictions(dl_model, train_loader_unshuffled)
    test_dl_preds, _, test_attentions = get_dl_predictions(dl_model, test_loader)
    
    met_feature_indices = [6, 7, 8, 9, 10]
    train_met_features = X_train[:, -1, met_feature_indices]
    test_met_features = X_test[:, -1, met_feature_indices]
    
    xgb_wrapper = CBLA_XGBoost(max_depth=6, n_estimators=100)
    
    X_train_xgb = xgb_wrapper.prepare_features(train_dl_preds, train_met_features)
    X_test_xgb = xgb_wrapper.prepare_features(test_dl_preds, test_met_features)
    
    print("\n--- 4. Huan luyen mo hinh thu cap XGBoost ---")
    xgb_wrapper.fit(X_train_xgb, y_train)
    print("Huấn luyện XGBoost hoàn tất.")
    
    print("\n--- 5. Danh gia chi so mo hinh (Evaluation) ---")
    dl_metrics = evaluate_metrics(y_test, test_dl_preds)
    print_evaluation(dl_metrics, "Deep Learning doc lap (CNN-BiLSTM-Attention)")
    
    final_preds = xgb_wrapper.predict(X_test_xgb)
    cbla_metrics = evaluate_metrics(y_test, final_preds)
    print_evaluation(cbla_metrics, "Toan bo mo hinh lai CBLA (Hybrid)")
    
    print("\n--- 6. Xuat bieu do va luu trong so---")
    plot_results(losses, y_test, final_preds, test_attentions, xgb_wrapper)
    
    # Save models
    os.makedirs("saved_models", exist_ok=True)
    torch.save(dl_model.state_dict(), "saved_models/cbla_dl_weights.pth")
    xgb_wrapper.model.save_model("saved_models/cbla_xgboost.json")
    print("[INFO] Da luu trong so mo hinh tai thu muc 'saved_models/'.")
    print("\nHoan tat toan bo quy trinh. Mo hinh da san sang.")

if __name__ == "__main__":
    main()
