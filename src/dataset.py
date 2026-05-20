import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(data, time_steps=24, target_col_idx=0):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, target_col_idx])
    return np.array(X), np.array(y)

def preprocess_data(csv_path="data/beijing_air_quality_2014.csv", time_steps=24):
    df = pd.read_csv(csv_path)
    
    # We only use single station for simplicity (Station_id and Date are not used in DL)
    features_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Temp', 'Humid', 'WindSpeed', 'WindDir', 'Weather']
    df_features = df[features_cols].copy()
    
    # 1. Fill missing values with KNN-Imputer
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(df_features)
    
    # 2. Normalize with Min-Max Scaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    # PM2.5 is at index 0
    target_idx = 0
    
    # Create sliding windows
    X, y = create_sliding_windows(data_scaled, time_steps=time_steps, target_col_idx=target_idx)
    
    # Train-test split (80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    train_loader_unshuffled = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # Return loaders and the scaler to inverse-transform later if needed
    return train_loader, test_loader, train_loader_unshuffled, scaler, X_train, y_train, X_test, y_test
