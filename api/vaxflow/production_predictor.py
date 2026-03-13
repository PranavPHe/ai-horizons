"""
Production Yield Predictor for VaxFlow

ML model to predict vaccine batch yields based on production parameters.
Supports multiple model types: Random Forest, Gradient Boosting, and Neural Networks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)


@dataclass
class BatchParameters:
    """Parameters for a vaccine production batch."""
    temperature: float  # Incubation temperature (°C)
    humidity: float  # Relative humidity (%)
    incubation_time: float  # Hours
    cell_density: float  # Cells per mL (x10^6)
    nutrient_concentration: float  # Nutrient media concentration (g/L)
    ph_level: float  # pH of culture medium
    dissolved_oxygen: float  # Dissolved oxygen percentage
    
    # Optional parameters
    batch_size_liters: float = 1000.0
    cell_line: str = "HEK293"
    passage_number: int = 10
    
    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "incubation_time": self.incubation_time,
            "cell_density": self.cell_density,
            "nutrient_concentration": self.nutrient_concentration,
            "ph_level": self.ph_level,
            "dissolved_oxygen": self.dissolved_oxygen,
            "batch_size_liters": self.batch_size_liters,
            "passage_number": self.passage_number
        }
    
    def to_array(self, feature_columns: List[str]) -> np.ndarray:
        """Convert to numpy array with specified feature order."""
        data = self.to_dict()
        return np.array([data[col] for col in feature_columns])


@dataclass
class ProductionMetrics:
    """Metrics from production prediction model."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared score
    cv_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        return (
            f"Production Model Metrics:\n"
            f"  MAE:  {self.mae:,.0f} doses\n"
            f"  RMSE: {self.rmse:,.0f} doses\n"
            f"  R²:   {self.r2:.4f}\n"
            f"  CV Mean: {np.mean(self.cv_scores):.4f} (±{np.std(self.cv_scores):.4f})"
        )


class ProductionPredictor:
    """
    ML model to predict vaccine production batch yields.
    
    Example usage:
    ```python
    predictor = ProductionPredictor(model_type="gradient_boosting")
    predictor.fit(production_data, feature_columns, target_column="yield")
    
    # Predict for new batches
    new_batch = BatchParameters(
        temperature=37.0,
        humidity=85.0,
        incubation_time=72.0,
        cell_density=2.5,
        nutrient_concentration=15.0,
        ph_level=7.2,
        dissolved_oxygen=40.0
    )
    predicted_yield = predictor.predict_single(new_batch)
    ```
    """
    
    SUPPORTED_MODELS = ["random_forest", "gradient_boosting", "neural_net"]
    
    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
        **model_params
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column: str = ""
        self.metrics: Optional[ProductionMetrics] = None
        self._is_fitted = False
    
    def _create_model(self):
        """Create the underlying ML model based on model_type."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": -1
            }
            default_params.update(self.model_params)
            return RandomForestRegressor(**default_params)
        
        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "min_samples_split": 5,
                "random_state": self.random_state
            }
            default_params.update(self.model_params)
            return GradientBoostingRegressor(**default_params)
        
        elif self.model_type == "neural_net":
            # Use sklearn's MLPRegressor for simplicity
            from sklearn.neural_network import MLPRegressor
            default_params = {
                "hidden_layer_sizes": (128, 64, 32),
                "activation": "relu",
                "solver": "adam",
                "learning_rate": "adaptive",
                "max_iter": 500,
                "random_state": self.random_state,
                "early_stopping": True,
                "validation_fraction": 0.1
            }
            default_params.update(self.model_params)
            return MLPRegressor(**default_params)
    
    def fit(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "yield",
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> "ProductionPredictor":
        """
        Fit the production predictor on historical batch data.
        
        Args:
            data: DataFrame with production batch data
            feature_columns: List of column names to use as features
            target_column: Name of the yield/output column
            test_size: Fraction of data to hold out for testing
            cv_folds: Number of cross-validation folds
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ProductionPredictor with {len(data)} samples...")
        
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Prepare features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Cross-validation on full training data
        X_scaled = self.scaler.transform(X)
        cv_scores = cross_val_score(
            self._create_model(), X_scaled, y, cv=cv_folds, scoring="r2"
        )
        
        # Calculate feature importance
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, importance in zip(feature_columns, self.model.feature_importances_):
                feature_importance[name] = float(importance)
        
        self.metrics = ProductionMetrics(
            mae=mean_absolute_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            r2=r2_score(y_test, y_pred),
            cv_scores=cv_scores.tolist(),
            feature_importance=feature_importance
        )
        
        self._is_fitted = True
        logger.info(f"Model fitted. R² = {self.metrics.r2:.4f}")
        
        return self
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict yields for multiple batches.
        
        Args:
            data: DataFrame or array with batch parameters
        
        Returns:
            Array of predicted yields
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if isinstance(data, pd.DataFrame):
            X = data[self.feature_columns].values
        else:
            X = data
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_single(self, batch: BatchParameters) -> float:
        """Predict yield for a single batch."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = batch.to_array(self.feature_columns).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])
    
    def get_optimal_parameters(
        self,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Find optimal production parameters to maximize yield.
        
        Uses random search within constraints to find best parameters.
        
        Args:
            constraints: Dict of {feature: (min, max)} constraints
            n_samples: Number of random samples to evaluate
        
        Returns:
            Dict of optimal parameter values
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before optimization")
        
        # Default constraints based on typical vaccine production
        default_constraints = {
            "temperature": (35.0, 39.0),
            "humidity": (70.0, 95.0),
            "incubation_time": (48.0, 96.0),
            "cell_density": (1.0, 5.0),
            "nutrient_concentration": (10.0, 25.0),
            "ph_level": (6.8, 7.6),
            "dissolved_oxygen": (30.0, 60.0)
        }
        
        constraints = {**default_constraints, **(constraints or {})}
        
        # Generate random samples
        samples = np.zeros((n_samples, len(self.feature_columns)))
        for i, col in enumerate(self.feature_columns):
            if col in constraints:
                low, high = constraints[col]
                samples[:, i] = np.random.uniform(low, high, n_samples)
            else:
                # Use training data range if no constraint specified
                samples[:, i] = np.random.uniform(0, 100, n_samples)
        
        # Predict and find best
        predictions = self.predict(samples)
        best_idx = np.argmax(predictions)
        
        optimal = {
            col: float(samples[best_idx, i])
            for i, col in enumerate(self.feature_columns)
        }
        optimal["predicted_yield"] = float(predictions[best_idx])
        
        logger.info(f"Optimal parameters found with predicted yield: {optimal['predicted_yield']:,.0f}")
        
        return optimal
    
    def get_sensitivity_analysis(
        self,
        base_params: BatchParameters,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on parameters.
        
        Args:
            base_params: Baseline parameter values
            param_ranges: Ranges to vary each parameter over
            n_points: Number of points per parameter
        
        Returns:
            DataFrame with sensitivity analysis results
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before analysis")
        
        base_values = base_params.to_dict()
        
        default_ranges = {
            "temperature": (34.0, 40.0),
            "humidity": (60.0, 100.0),
            "incubation_time": (24.0, 120.0),
            "cell_density": (0.5, 6.0),
            "nutrient_concentration": (5.0, 30.0),
            "ph_level": (6.5, 8.0),
            "dissolved_oxygen": (20.0, 70.0)
        }
        param_ranges = {**default_ranges, **(param_ranges or {})}
        
        results = []
        for param in self.feature_columns:
            if param not in param_ranges:
                continue
            
            low, high = param_ranges[param]
            values = np.linspace(low, high, n_points)
            
            for val in values:
                test_params = base_values.copy()
                test_params[param] = val
                X = np.array([[test_params[col] for col in self.feature_columns]])
                X_scaled = self.scaler.transform(X)
                pred = self.model.predict(X_scaled)[0]
                
                results.append({
                    "parameter": param,
                    "value": val,
                    "predicted_yield": pred
                })
        
        return pd.DataFrame(results)
    
    def save(self, filepath: str):
        """Save the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "model_type": self.model_type,
            "metrics": self.metrics
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ProductionPredictor":
        """Load a fitted model from disk."""
        data = joblib.load(filepath)
        predictor = cls(model_type=data["model_type"])
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor.feature_columns = data["feature_columns"]
        predictor.target_column = data["target_column"]
        predictor.metrics = data["metrics"]
        predictor._is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return predictor
