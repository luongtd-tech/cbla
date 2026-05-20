import xgboost as xgb
import numpy as np

class CBLA_XGBoost:
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        # Using parameters similar to the paper
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='reg:squarederror' # equivalent to regression
        )
        
    def prepare_features(self, dl_predictions, meteorological_features):
        """
        Combines the Deep Learning preliminary prediction with meteorological features.
        dl_predictions: (N,) array
        meteorological_features: (N, num_met_features) array. E.g. Temp, Humid, WindSpeed, WindDir, Weather
        """
        # Ensure correct shapes
        if len(dl_predictions.shape) == 1:
            dl_predictions = dl_predictions.reshape(-1, 1)
            
        # Concatenate dl_predictions as an additional feature to the meteorological features
        # Assuming meteorological_features are at the current timestep (t)
        combined_features = np.hstack((dl_predictions, meteorological_features))
        return combined_features

    def fit(self, X_combined, y):
        self.model.fit(X_combined, y)
        
    def predict(self, X_combined):
        return self.model.predict(X_combined)
