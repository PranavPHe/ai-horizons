"""
VaxFlow - Vaccine Production Efficiency and Distribution Optimization

ML-powered platform for:
- Production yield prediction
- Demand forecasting
- Distribution optimization
"""

__version__ = "0.1.0"
__author__ = "AI Horizons"

from .production_predictor import ProductionPredictor, BatchParameters
from .demand_forecaster import DemandForecaster, ForecastResult
from .distribution_optimizer import DistributionOptimizer, AllocationPlan
from .pipeline import VaxFlowPipeline
from .analyzer import (
    VaxFlowAnalyzer,
    FacilityInput,
    EnvironmentInput,
    ShippingLocation,
    OptimizationResult,
    analyze_from_data
)

__all__ = [
    "ProductionPredictor",
    "BatchParameters", 
    "DemandForecaster",
    "ForecastResult",
    "DistributionOptimizer",
    "AllocationPlan",
    "VaxFlowPipeline",
    "VaxFlowAnalyzer",
    "FacilityInput",
    "EnvironmentInput",
    "ShippingLocation",
    "OptimizationResult",
    "analyze_from_data",
]
