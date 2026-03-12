"""
Demand Forecasting Module for VaxFlow

Time-series forecasting for vaccine demand prediction.
Supports ARIMA, Exponential Smoothing, and LSTM models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from demand forecasting."""
    dates: List[datetime]
    predictions: np.ndarray
    lower_bound: np.ndarray  # Lower confidence interval
    upper_bound: np.ndarray  # Upper confidence interval
    confidence_level: float = 0.95
    region: Optional[str] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "demand": self.predictions,
            "lower_ci": self.lower_bound,
            "upper_ci": self.upper_bound,
            "region": self.region
        })
    
    @property
    def total_demand(self) -> float:
        return float(np.sum(self.predictions))
    
    @property
    def mean_daily_demand(self) -> float:
        return float(np.mean(self.predictions))


@dataclass 
class ForecastMetrics:
    """Metrics for forecast model evaluation."""
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    coverage: float  # Prediction interval coverage
    
    def summary(self) -> str:
        return (
            f"Forecast Model Metrics:\n"
            f"  MAE:  {self.mae:,.0f} doses\n"
            f"  RMSE: {self.rmse:,.0f} doses\n"
            f"  MAPE: {self.mape:.2f}%\n"
            f"  CI Coverage: {self.coverage*100:.1f}%"
        )


class DemandForecaster:
    """
    Time-series forecasting for vaccine demand.
    
    Example usage:
    ```python
    forecaster = DemandForecaster(model_type="arima")
    forecaster.fit(historical_demand)
    
    forecast = forecaster.forecast(horizon=30)
    print(f"Total forecasted demand: {forecast.total_demand:,.0f} doses")
    ```
    """
    
    SUPPORTED_MODELS = ["arima", "exponential_smoothing", "lstm"]
    
    def __init__(
        self,
        model_type: str = "arima",
        seasonality_period: int = 7,  # Weekly seasonality
        confidence_level: float = 0.95,
        **model_params
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.seasonality_period = seasonality_period
        self.confidence_level = confidence_level
        self.model_params = model_params
        
        self.models: Dict[str, object] = {}  # One model per region
        self.history: Dict[str, pd.Series] = {}
        self.metrics: Dict[str, ForecastMetrics] = {}
        self._is_fitted = False
    
    def fit(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        demand_column: str = "demand",
        region_column: Optional[str] = "region",
        validation_size: int = 14  # Days for validation
    ) -> "DemandForecaster":
        """
        Fit the forecasting model on historical demand data.
        
        Args:
            data: DataFrame with demand history
            date_column: Name of date column
            demand_column: Name of demand column
            region_column: Name of region column (if multi-region)
            validation_size: Number of days to hold out for validation
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting DemandForecaster on {len(data)} records...")
        
        # Ensure date column is datetime
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Get regions or use 'all' if no region column
        if region_column and region_column in data.columns:
            regions = data[region_column].unique()
        else:
            regions = ["all"]
            region_column = None
        
        for region in regions:
            region_str = str(region)
            
            if region_column:
                region_data = data[data[region_column] == region].copy()
            else:
                region_data = data.copy()
            
            # Sort by date and create time series
            region_data = region_data.sort_values(date_column)
            ts = region_data.set_index(date_column)[demand_column]
            
            # Resample to daily and fill gaps
            ts = ts.resample('D').sum().fillna(method='ffill').fillna(0)
            
            # Store history
            self.history[region_str] = ts
            
            # Split for validation
            train = ts.iloc[:-validation_size] if validation_size > 0 else ts
            
            # Fit model based on type
            if self.model_type == "arima":
                model = self._fit_arima(train, region_str)
            elif self.model_type == "exponential_smoothing":
                model = self._fit_exp_smoothing(train, region_str)
            elif self.model_type == "lstm":
                model = self._fit_lstm(train, region_str)
            
            self.models[region_str] = model
            
            # Validate if we have validation data
            if validation_size > 0:
                val = ts.iloc[-validation_size:]
                val_pred = self._predict_internal(model, validation_size, train)
                self.metrics[region_str] = self._calculate_metrics(val.values, val_pred)
        
        self._is_fitted = True
        logger.info(f"Forecaster fitted for {len(regions)} region(s)")
        
        return self
    
    def _fit_arima(self, ts: pd.Series, region: str):
        """Fit ARIMA/SARIMAX model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Auto-detect ARIMA parameters using simple heuristics
        # In production, you'd use pmdarima for auto_arima
        p, d, q = 2, 1, 2
        seasonal_order = (1, 1, 1, self.seasonality_period)
        
        model = SARIMAX(
            ts,
            order=(p, d, q),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False)
        logger.info(f"ARIMA fitted for {region}, AIC: {fitted.aic:.2f}")
        return fitted
    
    def _fit_exp_smoothing(self, ts: pd.Series, region: str):
        """Fit Exponential Smoothing model."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(
            ts,
            seasonal_periods=self.seasonality_period,
            trend="add",
            seasonal="add",
            damped_trend=True
        )
        
        fitted = model.fit(optimized=True)
        logger.info(f"Exponential Smoothing fitted for {region}")
        return fitted
    
    def _fit_lstm(self, ts: pd.Series, region: str):
        """Fit LSTM model using sklearn's MLPRegressor as a simplified alternative."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler
        
        # Create sequences
        lookback = self.seasonality_period * 2
        X, y = self._create_sequences(ts.values, lookback)
        
        # Scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and fit MLP (simulating LSTM behavior)
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True
        )
        mlp.fit(X_scaled, y)
        
        logger.info(f"LSTM (MLP) fitted for {region}")
        return {"model": mlp, "scaler": scaler, "lookback": lookback, "last_values": ts.values[-lookback:]}
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _predict_internal(self, model, horizon: int, history: pd.Series) -> np.ndarray:
        """Internal prediction using fitted model."""
        if self.model_type == "arima":
            forecast = model.get_forecast(steps=horizon)
            return forecast.predicted_mean.values
        
        elif self.model_type == "exponential_smoothing":
            return model.forecast(horizon).values
        
        elif self.model_type == "lstm":
            predictions = []
            current_sequence = model["last_values"].copy()
            scaler = model["scaler"]
            mlp = model["model"]
            lookback = model["lookback"]
            
            for _ in range(horizon):
                X = scaler.transform(current_sequence.reshape(1, -1))
                pred = mlp.predict(X)[0]
                predictions.append(pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred
            
            return np.array(predictions)
    
    def _get_confidence_intervals(
        self,
        model,
        horizon: int,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for predictions."""
        if self.model_type == "arima":
            forecast = model.get_forecast(steps=horizon)
            conf_int = forecast.conf_int(alpha=1 - self.confidence_level)
            return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
        
        else:
            # Estimate confidence intervals based on historical variance
            std = np.std(predictions) * 0.2  # Simplified estimation
            z = 1.96  # For 95% CI
            lower = predictions - z * std * np.sqrt(np.arange(1, horizon + 1))
            upper = predictions + z * std * np.sqrt(np.arange(1, horizon + 1))
            return np.maximum(0, lower), upper
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
        """Calculate forecast evaluation metrics."""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
        
        # Coverage (assuming 95% CI covers +/- 2 * RMSE)
        residuals = np.abs(actual - predicted)
        coverage = np.mean(residuals <= 2 * rmse)
        
        return ForecastMetrics(mae=mae, rmse=rmse, mape=mape, coverage=coverage)
    
    def forecast(
        self,
        horizon: int = 30,
        regions: Optional[List[str]] = None,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate demand forecasts.
        
        Args:
            horizon: Number of days to forecast
            regions: List of regions to forecast (None = all)
            start_date: Start date for forecast (default = day after last observation)
        
        Returns:
            DataFrame with forecasts for all regions
        """
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before forecasting")
        
        regions = regions or list(self.models.keys())
        results = []
        
        for region in regions:
            region_str = str(region)
            if region_str not in self.models:
                logger.warning(f"No model for region {region}, skipping")
                continue
            
            model = self.models[region_str]
            history = self.history[region_str]
            
            # Determine start date
            if start_date is None:
                forecast_start = history.index[-1] + timedelta(days=1)
            else:
                forecast_start = start_date
            
            # Generate dates
            dates = [forecast_start + timedelta(days=i) for i in range(horizon)]
            
            # Predict
            predictions = self._predict_internal(model, horizon, history)
            lower, upper = self._get_confidence_intervals(model, horizon, predictions)
            
            # Ensure non-negative
            predictions = np.maximum(0, predictions)
            lower = np.maximum(0, lower)
            
            result = ForecastResult(
                dates=dates,
                predictions=predictions,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=self.confidence_level,
                region=region_str
            )
            
            results.append(result.to_dataframe())
        
        combined = pd.concat(results, ignore_index=True)
        logger.info(f"Generated {horizon}-day forecast for {len(regions)} region(s)")
        
        return combined
    
    def forecast_single_region(
        self,
        region: str,
        horizon: int = 30,
        start_date: Optional[datetime] = None
    ) -> ForecastResult:
        """Get forecast for a single region."""
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before forecasting")
        
        region_str = str(region)
        if region_str not in self.models:
            raise ValueError(f"No model fitted for region: {region}")
        
        model = self.models[region_str]
        history = self.history[region_str]
        
        if start_date is None:
            forecast_start = history.index[-1] + timedelta(days=1)
        else:
            forecast_start = start_date
        
        dates = [forecast_start + timedelta(days=i) for i in range(horizon)]
        predictions = self._predict_internal(model, horizon, history)
        lower, upper = self._get_confidence_intervals(model, horizon, predictions)
        
        predictions = np.maximum(0, predictions)
        lower = np.maximum(0, lower)
        
        return ForecastResult(
            dates=dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.confidence_level,
            region=region_str
        )
    
    def get_seasonal_patterns(self, region: str = "all") -> Dict[str, float]:
        """Analyze seasonal patterns in demand."""
        if region not in self.history:
            raise ValueError(f"No history for region: {region}")
        
        ts = self.history[region]
        
        # Calculate day-of-week effects
        ts_with_dow = pd.DataFrame({"demand": ts, "dow": ts.index.dayofweek})
        dow_means = ts_with_dow.groupby("dow")["demand"].mean()
        overall_mean = ts.mean()
        
        dow_effects = {
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i]: 
            float(dow_means.get(i, overall_mean) / overall_mean - 1)
            for i in range(7)
        }
        
        return dow_effects
    
    def save(self, filepath: str):
        """Save fitted forecaster to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted forecaster")
        joblib.dump({
            "models": self.models,
            "history": self.history,
            "metrics": self.metrics,
            "model_type": self.model_type,
            "seasonality_period": self.seasonality_period,
            "confidence_level": self.confidence_level
        }, filepath)
        logger.info(f"Forecaster saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "DemandForecaster":
        """Load fitted forecaster from disk."""
        data = joblib.load(filepath)
        forecaster = cls(
            model_type=data["model_type"],
            seasonality_period=data["seasonality_period"],
            confidence_level=data["confidence_level"]
        )
        forecaster.models = data["models"]
        forecaster.history = data["history"]
        forecaster.metrics = data["metrics"]
        forecaster._is_fitted = True
        logger.info(f"Forecaster loaded from {filepath}")
        return forecaster
