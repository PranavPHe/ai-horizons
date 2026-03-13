#!/usr/bin/env python3
"""
VaxFlow - Vaccine Production Efficiency and Distribution Optimization

Main entry point demonstrating the complete ML pipeline.

Usage:
    python main.py                     # Run full demo
    python main.py --interactive       # Interactive analysis mode
    python main.py --analyze           # Quick analysis with sample data
    python main.py --component prod    # Run production prediction only
    python main.py --component demand  # Run demand forecasting only
    python main.py --component dist    # Run distribution optimization only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import VaxFlowPipeline, PipelineConfig
from production_predictor import ProductionPredictor, BatchParameters
from demand_forecaster import DemandForecaster
from distribution_optimizer import DistributionOptimizer
from sample_data import generate_complete_dataset
from analyzer import (
    VaxFlowAnalyzer, 
    FacilityInput, 
    EnvironmentInput, 
    ShippingLocation,
    interactive_session,
    analyze_from_data
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VaxFlow")


def demo_production_prediction():
    """Demonstrate production yield prediction."""
    print("\n" + "="*60)
    print("PRODUCTION YIELD PREDICTION")
    print("="*60)
    
    # Generate sample data
    from sample_data import generate_production_data
    data = generate_production_data(n_batches=500)
    
    print(f"Training data: {len(data)} batches")
    print(f"Yield range: {data['yield'].min():,.0f} - {data['yield'].max():,.0f} doses")
    
    # Train model
    predictor = ProductionPredictor(model_type="gradient_boosting")
    feature_cols = [
        "temperature", "humidity", "incubation_time", "cell_density",
        "nutrient_concentration", "ph_level", "dissolved_oxygen"
    ]
    
    predictor.fit(data, feature_columns=feature_cols, target_column="yield")
    
    print(f"\n{predictor.metrics.summary()}")
    
    # Feature importance
    if predictor.metrics.feature_importance:
        print("\nFeature Importance:")
        for name, imp in sorted(
            predictor.metrics.feature_importance.items(),
            key=lambda x: -x[1]
        ):
            print(f"  {name}: {imp:.3f}")
    
    # Predict for a new batch
    print("\n--- New Batch Prediction ---")
    new_batch = BatchParameters(
        temperature=37.0,
        humidity=85.0,
        incubation_time=72.0,
        cell_density=2.5,
        nutrient_concentration=15.0,
        ph_level=7.2,
        dissolved_oxygen=45.0
    )
    
    predicted_yield = predictor.predict_single(new_batch)
    print(f"Parameters: temp=37°C, humidity=85%, time=72h")
    print(f"Predicted Yield: {predicted_yield:,.0f} doses")
    
    # Find optimal parameters
    print("\n--- Optimal Parameters Search ---")
    optimal = predictor.get_optimal_parameters(n_samples=5000)
    print("Optimal production parameters:")
    for param, value in optimal.items():
        if param != "predicted_yield":
            print(f"  {param}: {value:.2f}")
    print(f"Expected Yield: {optimal['predicted_yield']:,.0f} doses")
    
    return predictor


def demo_demand_forecasting():
    """Demonstrate demand forecasting."""
    print("\n" + "="*60)
    print("DEMAND FORECASTING")
    print("="*60)
    
    # Generate history
    from sample_data import generate_demand_history
    history = generate_demand_history(n_days=365, n_regions=3)
    
    print(f"Historical data: {len(history)} records")
    print(f"Regions: {history['region'].unique().tolist()}")
    print(f"Date range: {history['date'].min()} to {history['date'].max()}")
    
    # Train forecaster
    forecaster = DemandForecaster(model_type="exponential_smoothing", seasonality_period=7)
    forecaster.fit(history, date_column="date", demand_column="demand", region_column="region")
    
    # Show metrics for each region
    print("\nModel Performance by Region:")
    for region, metrics in forecaster.metrics.items():
        print(f"  {region}: MAE={metrics.mae:,.0f}, MAPE={metrics.mape:.1f}%")
    
    # Generate forecast
    print("\n--- 30-Day Forecast ---")
    forecast = forecaster.forecast(horizon=30)
    
    # Summarize by region
    summary = forecast.groupby("region")["demand"].agg(["sum", "mean", "std"])
    print("\nForecast Summary:")
    print(summary.round(0))
    
    total_forecast = forecast["demand"].sum()
    print(f"\nTotal 30-day forecasted demand: {total_forecast:,.0f} doses")
    
    # Seasonal patterns
    print("\n--- Seasonal Patterns (Day-of-Week Effects) ---")
    patterns = forecaster.get_seasonal_patterns("North")
    for day, effect in patterns.items():
        direction = "+" if effect >= 0 else ""
        print(f"  {day}: {direction}{effect*100:.1f}%")
    
    return forecaster


def demo_distribution_optimization():
    """Demonstrate distribution optimization."""
    print("\n" + "="*60)
    print("DISTRIBUTION OPTIMIZATION")
    print("="*60)
    
    from sample_data import generate_facility_data, generate_region_data
    
    facilities = generate_facility_data(n_facilities=3)
    regions = generate_region_data(n_regions=4)
    
    print("Facilities:")
    print(facilities[["id", "name", "capacity", "inventory"]].to_string(index=False))
    
    print("\nRegions:")
    print(regions[["id", "name", "population", "coverage"]].to_string(index=False))
    
    # Create optimizer
    optimizer = DistributionOptimizer(
        objective="minimize_cost",
        max_transport_time=48.0
    )
    optimizer.set_facilities(facilities)
    optimizer.set_regions(regions)
    
    # Define supply and demand
    supply = {row["id"]: row["inventory"] for _, row in facilities.iterrows()}
    demand = {row["id"]: row["population"] * 0.05 for _, row in regions.iterrows()}  # 5% of pop
    
    print(f"\nTotal Supply: {sum(supply.values()):,.0f} doses")
    print(f"Total Demand: {sum(demand.values()):,.0f} doses")
    
    # Optimize
    plan = optimizer.optimize(supply, demand)
    
    print("\n--- Optimized Distribution Plan ---")
    print(plan.summary())
    
    # Show transport details
    if plan.transport_details:
        print("\nTransport Routes:")
        for route in plan.transport_details[:5]:  # Show top 5
            print(f"  {route['from_facility']} → {route['to_region']}: "
                  f"{route['doses']:,.0f} doses, {route['distance_km']:.0f} km, "
                  f"${route['cost']:.2f}")
    
    # Scenario analysis
    print("\n--- Scenario Analysis ---")
    scenarios = optimizer.simulate_scenarios(supply, demand)
    print(scenarios.to_string(index=False))
    
    return optimizer


def demo_full_pipeline():
    """Demonstrate the complete VaxFlow pipeline."""
    print("\n" + "="*60)
    print("COMPLETE VAXFLOW PIPELINE")
    print("="*60)
    
    # Generate all data
    datasets = generate_complete_dataset()
    
    # Configure pipeline
    config = PipelineConfig(
        production_model_type="gradient_boosting",
        forecast_horizon=30,
        forecast_model_type="exponential_smoothing",
        optimization_objective="minimize_cost"
    )
    
    print("Pipeline Configuration:")
    print(f"  Production Model: {config.production_model_type}")
    print(f"  Forecast Model: {config.forecast_model_type}")
    print(f"  Forecast Horizon: {config.forecast_horizon} days")
    print(f"  Optimization: {config.optimization_objective}")
    
    # Create and fit pipeline
    pipeline = VaxFlowPipeline(config)
    
    print("\nFitting pipeline on historical data...")
    pipeline.fit(
        production_data=datasets["production"],
        demand_history=datasets["demand"],
        facility_data=datasets["facilities"],
        region_data=datasets["regions"]
    )
    
    # Prepare upcoming batch parameters
    upcoming_batches = pd.DataFrame({
        "temperature": [37.0, 37.2, 36.8, 37.1, 36.9],
        "humidity": [85, 84, 86, 85, 85],
        "incubation_time": [72, 70, 74, 72, 72],
        "cell_density": [2.5, 2.4, 2.6, 2.5, 2.5],
        "nutrient_concentration": [15, 14, 16, 15, 15],
        "ph_level": [7.2, 7.1, 7.3, 7.2, 7.2],
        "dissolved_oxygen": [45, 44, 46, 45, 45]
    })
    
    # Run pipeline
    print("\nRunning optimization pipeline...")
    result = pipeline.run(
        batch_params=upcoming_batches,
        forecast_horizon=30,
        regions=["North", "South", "East", "West", "Central"]
    )
    
    # Display results
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    
    print("\n1. PRODUCTION PREDICTIONS")
    print("-" * 40)
    print(f"  Total Predicted Doses: {result.production_predictions['total_predicted_doses']:,.0f}")
    print(f"  Mean Yield per Batch: {result.production_predictions['mean_yield_per_batch']:,.0f}")
    print(f"  Yield Std Dev: {result.production_predictions['yield_std']:,.0f}")
    
    print("\n2. DEMAND FORECAST (30 days)")
    print("-" * 40)
    if "region" in result.demand_forecast.columns:
        demand_by_region = result.demand_forecast.groupby("region")["demand"].sum()
        for region, demand in demand_by_region.items():
            print(f"  {region}: {demand:,.0f} doses")
    print(f"  Total: {result.demand_forecast['demand'].sum():,.0f} doses")
    
    print("\n3. DISTRIBUTION PLAN")
    print("-" * 40)
    print(f"  Total Cost: ${result.distribution_plan.get('total_cost', 0):,.2f}")
    print(f"  Doses Distributed: {result.distribution_plan.get('total_doses_distributed', 0):,.0f}")
    print(f"  Unmet Demand: {result.distribution_plan.get('unmet_demand', 0):,.0f}")
    
    print("\n4. KEY METRICS")
    print("-" * 40)
    for metric, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n5. RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_file = "vaxflow_results.json"
    result.to_json(output_file)
    print(f"\nResults saved to {output_file}")
    
    return pipeline, result


def demo_quick_analysis():
    """
    Demonstrate quick analysis with user-provided parameters.
    Shows what the AI does when given production and distribution inputs.
    """
    print("\n" + "="*60)
    print("QUICK ANALYSIS MODE")
    print("="*60)
    print("\nAnalyzing production parameters and optimizing distribution...")
    
    analyzer = VaxFlowAnalyzer()
    
    # Example: Add production facilities with bioreactor capacity
    print("\n📍 Setting up production facilities...")
    facilities = [
        FacilityInput(
            facility_id="PLANT_NE",
            name="Northeast Production Plant",
            bioreactor_capacity_liters=10000,
            current_inventory=750000,
            latitude=40.7,
            longitude=-74.0
        ),
        FacilityInput(
            facility_id="PLANT_SE",
            name="Southeast Production Center",
            bioreactor_capacity_liters=8000,
            current_inventory=500000,
            latitude=33.7,
            longitude=-84.4
        ),
        FacilityInput(
            facility_id="PLANT_W",
            name="West Coast Facility",
            bioreactor_capacity_liters=12000,
            current_inventory=900000,
            latitude=37.8,
            longitude=-122.4
        )
    ]
    
    for f in facilities:
        analyzer.add_facility(f)
        print(f"  ✓ {f.name}: {f.bioreactor_capacity_liters}L capacity, {f.current_inventory:,} doses inventory")
    
    # Example: Set current environmental parameters (suboptimal to show recommendations)
    print("\n🌡️  Current Environmental Parameters:")
    env = EnvironmentInput(
        temperature=36.2,      # Slightly low (optimal: 37)
        ph_level=7.4,          # Slightly high (optimal: 7.2)
        nutrient_concentration=12.5,  # Low (optimal: 15)
        humidity=82.0,
        dissolved_oxygen=42.0,
        incubation_time=68.0,
        cell_density=2.3
    )
    analyzer.set_environment(env)
    
    print(f"  Temperature: {env.temperature}°C")
    print(f"  pH Level: {env.ph_level}")
    print(f"  Nutrient Concentration: {env.nutrient_concentration} g/L")
    print(f"  Humidity: {env.humidity}%")
    print(f"  Dissolved Oxygen: {env.dissolved_oxygen}%")
    print(f"  Incubation Time: {env.incubation_time} hours")
    print(f"  Cell Density: {env.cell_density} x10^6/mL")
    
    # Example: Add shipping locations (transportation routes)
    print("\n🚚 Shipping Locations:")
    locations = [
        ShippingLocation("NYC", "New York Metro", 40.7, -74.0, 20_000_000, priority=1.2),
        ShippingLocation("ATL", "Atlanta Region", 33.7, -84.4, 6_000_000, priority=1.0),
        ShippingLocation("CHI", "Chicago Area", 41.9, -87.6, 9_500_000, priority=1.1),
        ShippingLocation("LA", "Los Angeles", 34.1, -118.2, 13_000_000, priority=1.0),
        ShippingLocation("DAL", "Dallas-Fort Worth", 32.8, -96.8, 7_500_000, priority=0.9),
        ShippingLocation("MIA", "Miami Region", 25.8, -80.2, 6_200_000, priority=1.3),
    ]
    
    for loc in locations:
        analyzer.add_shipping_location(loc)
        print(f"  ✓ {loc.name}: Pop {loc.population:,}, Priority {loc.priority}")
    
    # Run the analysis
    print("\n" + "-"*60)
    print("🔬 RUNNING AI ANALYSIS...")
    print("-"*60)
    
    result = analyzer.analyze()
    
    # Display results
    print("\n" + result.summary())
    
    # Show detailed route information
    print("\n📦 OPTIMIZED DISTRIBUTION ROUTES (Minimizing Distance & Delays)")
    print("-" * 60)
    for i, route in enumerate(result.distribution_routes[:10], 1):
        print(f"  {i}. {route['from_name']} → {route['to_name']}")
        print(f"     Doses: {route['doses']:,.0f} | Distance: {route['distance_km']:.0f} km | "
              f"Time: {route['time_hours']:.1f} hrs | Cost: ${route['cost']:.2f}")
    
    if len(result.distribution_routes) > 10:
        print(f"  ... and {len(result.distribution_routes) - 10} more routes")
    
    # Export reports
    print("\n" + "-"*60)
    print("📄 EXPORTING LOGISTICS REPORTS...")
    print("-"*60)
    
    exports = result.export_all("vaxflow_report")
    print(f"  ✓ JSON Report:     {exports['json']}")
    print(f"  ✓ Routes CSV:      {exports['csv']}")
    print(f"  ✓ Full Report:     {exports['txt']}")
    
    print("\n  Reports exported successfully!")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="VaxFlow - Vaccine Production and Distribution ML Platform"
    )
    parser.add_argument(
        "--component", "-c",
        choices=["prod", "demand", "dist", "full"],
        default="full",
        help="Which component to demo (default: full pipeline)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive analysis session (enter your own data)"
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Run quick analysis with sample parameters"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("╔" + "═"*58 + "╗")
    print("║" + " VAXFLOW - Vaccine ML Optimization Platform ".center(58) + "║")
    print("║" + " Production Efficiency & Distribution ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    
    try:
        if args.interactive:
            interactive_session()
        elif args.analyze:
            demo_quick_analysis()
        elif args.component == "prod":
            demo_production_prediction()
        elif args.component == "demand":
            demo_demand_forecasting()
        elif args.component == "dist":
            demo_distribution_optimization()
        else:
            demo_full_pipeline()
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
