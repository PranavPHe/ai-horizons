"""
VaxFlow ML Pipeline

Orchestrates the complete vaccine production and distribution optimization workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the VaxFlow pipeline."""
    # Production settings
    production_model_type: str = "gradient_boosting"  # 'random_forest', 'gradient_boosting', 'neural_net'
    production_features: List[str] = field(default_factory=lambda: [
        "temperature", "humidity", "incubation_time", "cell_density",
        "nutrient_concentration", "ph_level", "dissolved_oxygen"
    ])
    
    # Forecasting settings
    forecast_horizon: int = 30  # days
    forecast_model_type: str = "arima"  # 'arima', 'prophet', 'lstm'
    seasonality_period: int = 7  # weekly seasonality
    
    # Distribution settings
    optimization_objective: str = "minimize_cost"  # 'minimize_cost', 'minimize_time', 'maximize_coverage'
    max_transport_time_hours: int = 48
    cold_chain_temperature: float = -70.0  # Celsius (for mRNA vaccines)
    
    # General settings
    random_state: int = 42
    verbose: bool = True


@dataclass
class PipelineResult:
    """Results from a complete pipeline run."""
    timestamp: datetime
    production_predictions: Dict[str, float]
    demand_forecast: pd.DataFrame
    distribution_plan: Dict[str, Any]
    metrics: Dict[str, float]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "production_predictions": self.production_predictions,
            "demand_forecast": self.demand_forecast.to_dict() if isinstance(self.demand_forecast, pd.DataFrame) else self.demand_forecast,
            "distribution_plan": self.distribution_plan,
            "metrics": self.metrics,
            "recommendations": self.recommendations
        }
    
    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class VaxFlowPipeline:
    """
    Main orchestration pipeline for vaccine production and distribution optimization.
    
    Example usage:
    ```python
    from vaxflow import VaxFlowPipeline, PipelineConfig
    
    config = PipelineConfig(
        forecast_horizon=30,
        optimization_objective="minimize_cost"
    )
    
    pipeline = VaxFlowPipeline(config)
    pipeline.fit(production_data, demand_history, facility_data)
    
    result = pipeline.optimize(
        target_date="2026-04-01",
        regions=["North", "South", "East", "West"]
    )
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.production_predictor = None
        self.demand_forecaster = None
        self.distribution_optimizer = None
        self._is_fitted = False
        
        if self.config.verbose:
            logger.info(f"VaxFlow Pipeline initialized with config: {self.config}")
    
    def fit(
        self,
        production_data: pd.DataFrame,
        demand_history: pd.DataFrame,
        facility_data: Optional[pd.DataFrame] = None,
        region_data: Optional[pd.DataFrame] = None
    ) -> "VaxFlowPipeline":
        """
        Fit all ML models in the pipeline.
        
        Args:
            production_data: Historical production batch data with features and yields
            demand_history: Time series of vaccine demand by region
            facility_data: Production facility information (capacity, location, etc.)
            region_data: Distribution region information (population, distance, etc.)
        
        Returns:
            Self for method chaining
        """
        from production_predictor import ProductionPredictor
        from demand_forecaster import DemandForecaster
        from distribution_optimizer import DistributionOptimizer
        
        logger.info("Fitting VaxFlow Pipeline...")
        
        # 1. Fit production yield predictor
        logger.info("Training production yield predictor...")
        self.production_predictor = ProductionPredictor(
            model_type=self.config.production_model_type,
            random_state=self.config.random_state
        )
        self.production_predictor.fit(
            production_data,
            feature_columns=self.config.production_features,
            target_column="yield"
        )
        
        # 2. Fit demand forecaster
        logger.info("Training demand forecaster...")
        self.demand_forecaster = DemandForecaster(
            model_type=self.config.forecast_model_type,
            seasonality_period=self.config.seasonality_period
        )
        self.demand_forecaster.fit(demand_history)
        
        # 3. Initialize distribution optimizer
        logger.info("Initializing distribution optimizer...")
        self.distribution_optimizer = DistributionOptimizer(
            objective=self.config.optimization_objective,
            max_transport_time=self.config.max_transport_time_hours,
            cold_chain_temp=self.config.cold_chain_temperature
        )
        if facility_data is not None:
            self.distribution_optimizer.set_facilities(facility_data)
        if region_data is not None:
            self.distribution_optimizer.set_regions(region_data)
        
        self._is_fitted = True
        logger.info("Pipeline fitting complete!")
        
        return self
    
    def predict_production(self, batch_params: pd.DataFrame) -> np.ndarray:
        """Predict production yields for given batch parameters."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before making predictions")
        return self.production_predictor.predict(batch_params)
    
    def forecast_demand(
        self,
        horizon: Optional[int] = None,
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Forecast vaccine demand for specified horizon and regions."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before forecasting")
        
        horizon = horizon or self.config.forecast_horizon
        return self.demand_forecaster.forecast(horizon=horizon, regions=regions)
    
    def optimize_distribution(
        self,
        available_supply: Dict[str, float],
        demand: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize vaccine distribution given supply and demand."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before optimizing")
        
        return self.distribution_optimizer.optimize(
            supply=available_supply,
            demand=demand,
            **kwargs
        )
    
    def run(
        self,
        batch_params: pd.DataFrame,
        forecast_horizon: Optional[int] = None,
        regions: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Run the complete pipeline: predict production, forecast demand, optimize distribution.
        
        Args:
            batch_params: Parameters for upcoming production batches
            forecast_horizon: Days to forecast ahead
            regions: Regions to include in optimization
        
        Returns:
            PipelineResult with predictions, forecasts, and recommendations
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before running")
        
        logger.info("Running complete VaxFlow pipeline...")
        
        # Step 1: Predict production yields
        yields = self.predict_production(batch_params)
        production_predictions = {
            "total_predicted_doses": float(yields.sum()),
            "batch_yields": yields.tolist(),
            "mean_yield_per_batch": float(yields.mean()),
            "yield_std": float(yields.std())
        }
        
        # Step 2: Forecast demand
        demand_forecast = self.forecast_demand(
            horizon=forecast_horizon or self.config.forecast_horizon,
            regions=regions
        )
        
        # Step 3: Optimize distribution based on predicted supply and forecasted demand
        total_supply = {"all": production_predictions["total_predicted_doses"]}
        total_demand = demand_forecast.groupby("region")["demand"].sum().to_dict() if "region" in demand_forecast.columns else {"all": demand_forecast["demand"].sum()}
        
        distribution_plan = self.optimize_distribution(
            available_supply=total_supply,
            demand=total_demand
        )
        
        # Step 4: Calculate metrics and generate recommendations
        metrics = self._calculate_metrics(
            production_predictions, demand_forecast, distribution_plan
        )
        recommendations = self._generate_recommendations(
            production_predictions, demand_forecast, distribution_plan, metrics
        )
        
        result = PipelineResult(
            timestamp=datetime.now(),
            production_predictions=production_predictions,
            demand_forecast=demand_forecast,
            distribution_plan=distribution_plan,
            metrics=metrics,
            recommendations=recommendations
        )
        
        logger.info("Pipeline run complete!")
        return result
    
    def _calculate_metrics(
        self,
        production: Dict,
        demand_df: pd.DataFrame,
        distribution: Dict
    ) -> Dict[str, float]:
        """Calculate pipeline performance metrics."""
        total_supply = production["total_predicted_doses"]
        total_demand = demand_df["demand"].sum() if "demand" in demand_df.columns else 0
        
        return {
            "supply_demand_ratio": total_supply / total_demand if total_demand > 0 else float('inf'),
            "coverage_rate": min(1.0, total_supply / total_demand) if total_demand > 0 else 1.0,
            "predicted_efficiency": production["mean_yield_per_batch"] / 1e6 if production["mean_yield_per_batch"] else 0,
            "distribution_cost": distribution.get("total_cost", 0),
            "wastage_estimate": max(0, total_supply - total_demand) if total_supply > total_demand else 0
        }
    
    def _generate_recommendations(
        self,
        production: Dict,
        demand_df: pd.DataFrame,
        distribution: Dict,
        metrics: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on pipeline outputs."""
        recommendations = []
        
        # Supply/demand balance recommendations
        ratio = metrics.get("supply_demand_ratio", 1.0)
        if ratio < 0.8:
            recommendations.append(
                f"WARNING: Predicted supply covers only {ratio*100:.1f}% of forecasted demand. "
                "Consider increasing production batches or optimizing yield parameters."
            )
        elif ratio > 1.5:
            recommendations.append(
                f"Predicted supply exceeds demand by {(ratio-1)*100:.1f}%. "
                "Consider reducing batch frequency to minimize wastage."
            )
        else:
            recommendations.append(
                "Supply and demand are well balanced for the forecast period."
            )
        
        # Production efficiency recommendations
        yield_std = production.get("yield_std", 0)
        mean_yield = production.get("mean_yield_per_batch", 1)
        cv = yield_std / mean_yield if mean_yield > 0 else 0
        
        if cv > 0.2:
            recommendations.append(
                f"High yield variability detected (CV={cv:.2f}). "
                "Review process parameters for consistency improvements."
            )
        
        # Distribution recommendations
        if distribution.get("unmet_demand", 0) > 0:
            recommendations.append(
                f"Distribution plan leaves {distribution['unmet_demand']:.0f} doses of demand unmet. "
                "Consider increasing production or prioritizing high-need regions."
            )
        
        return recommendations
    
    def save(self, filepath: str):
        """Save the fitted pipeline to disk."""
        import joblib
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")
        joblib.dump(self, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "VaxFlowPipeline":
        """Load a fitted pipeline from disk."""
        import joblib
        pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
