"""
VaxFlow Flask API Server

Provides REST endpoints for the web frontend to call the ML backend.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from production_predictor import ProductionPredictor, BatchParameters
from demand_forecaster import DemandForecaster
from distribution_optimizer import DistributionOptimizer
from sample_data import generate_production_data

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize and train the production predictor
production_model = None

def get_production_model():
    """Lazy-load and train the production model."""
    global production_model
    if production_model is None:
        data = generate_production_data(n_batches=500)
        production_model = ProductionPredictor(model_type="gradient_boosting")
        feature_cols = [
            "temperature", "humidity", "incubation_time", "cell_density",
            "nutrient_concentration", "ph_level", "dissolved_oxygen"
        ]
        production_model.fit(data, feature_columns=feature_cols, target_column="yield")
    return production_model


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "vaxflow-api"})


@app.route("/api/predict-yield", methods=["POST"])
def predict_yield():
    """
    Predict production batch yield using ML model.
    
    Expected JSON payload:
    {
        "temperature": 37.0,
        "humidity": 55,
        "incubation_time": 72,
        "cell_density": 2.5,
        "ph_level": 7.2,
        "dissolved_oxygen": 40,
        "nutrient_concentration": 15.0,
        "batch_size": 5000
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        params = BatchParameters(
            temperature=float(data.get("temperature", 37.0)),
            humidity=float(data.get("humidity", 55)),
            incubation_time=float(data.get("incubation_time", 72)),
            cell_density=float(data.get("cell_density", 2.5)),
            nutrient_concentration=float(data.get("nutrient_concentration", 15.0)),
            ph_level=float(data.get("ph_level", 7.2)),
            dissolved_oxygen=float(data.get("dissolved_oxygen", 40)),
            batch_size_liters=float(data.get("batch_size", 5000))
        )
        
        # Get model and predict
        model = get_production_model()
        predicted_yield, confidence = model.predict_single(params)
        
        # Calculate efficiency
        base_yield = params.batch_size_liters * 500  # ~500 doses per litre at 100%
        efficiency = min(1.0, predicted_yield / base_yield) if base_yield > 0 else 0.5
        
        # Quality rating
        if efficiency > 0.85:
            quality = "Excellent"
        elif efficiency > 0.7:
            quality = "Good"
        elif efficiency > 0.55:
            quality = "Acceptable"
        else:
            quality = "Below Target"
        
        # Feature importance
        importance = model.metrics.feature_importance if model.metrics else {}
        feature_importance = [
            {"name": k.replace("_", " ").title(), "importance": round(v, 3)}
            for k, v in sorted(importance.items(), key=lambda x: -x[1])
        ]
        
        # Generate suggestions
        suggestions = []
        if params.temperature < 35 or params.temperature > 39:
            suggestions.append({
                "text": f"Adjust temperature to 36–38°C (currently {params.temperature}°C)",
                "type": "warning"
            })
        if params.ph_level < 6.8 or params.ph_level > 7.6:
            suggestions.append({
                "text": f"Optimise pH to 6.8–7.6 (currently {params.ph_level})",
                "type": "warning"
            })
        if params.cell_density < 2:
            suggestions.append({
                "text": "Increase cell density above 2.0 ×10⁶/mL for better yield",
                "type": "info"
            })
        if params.dissolved_oxygen < 30 or params.dissolved_oxygen > 50:
            suggestions.append({
                "text": f"Target dissolved O₂ between 30–50% (currently {params.dissolved_oxygen}%)",
                "type": "info"
            })
        if not suggestions:
            suggestions.append({
                "text": "Parameters are well-optimised! No changes recommended.",
                "type": "success"
            })
        
        # Sensitivity analysis
        temps = [30, 32, 34, 35, 36, 37, 38, 39, 40, 42, 44]
        sensitivity = []
        for t in temps:
            test_params = BatchParameters(
                temperature=t,
                humidity=params.humidity,
                incubation_time=params.incubation_time,
                cell_density=params.cell_density,
                nutrient_concentration=params.nutrient_concentration,
                ph_level=params.ph_level,
                dissolved_oxygen=params.dissolved_oxygen,
                batch_size_liters=params.batch_size_liters
            )
            y, _ = model.predict_single(test_params)
            sensitivity.append({"temperature": t, "yield": int(y)})
        
        return jsonify({
            "success": True,
            "prediction": {
                "yield": int(predicted_yield),
                "efficiency": round(efficiency * 100, 1),
                "quality": quality,
                "confidence": round(confidence, 3) if confidence else None
            },
            "feature_importance": feature_importance,
            "suggestions": suggestions,
            "sensitivity": sensitivity,
            "model_metrics": {
                "mae": round(model.metrics.mae, 0) if model.metrics else None,
                "r2": round(model.metrics.r2, 4) if model.metrics else None
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/forecast-demand", methods=["POST"])
def forecast_demand():
    """
    Forecast vaccine demand across regions.
    
    Expected JSON payload:
    {
        "days": 30,
        "regions": 5,
        "base_demand": 50000,
        "growth_rate": 5.0
    }
    """
    try:
        data = request.get_json()
        
        days = int(data.get("days", 30))
        num_regions = int(data.get("regions", 5))
        base_demand = int(data.get("base_demand", 50000))
        growth_rate = float(data.get("growth_rate", 5.0)) / 100
        
        region_names = [
            "North", "South", "East", "West", "Central",
            "Northeast", "Southeast", "Northwest", "Southwest", "Midwest"
        ][:num_regions]
        
        # Generate forecasts for each region
        regions = []
        all_forecasts = []
        
        for i, name in enumerate(region_names):
            # Regional variation factor
            factor = 0.6 + np.random.random() * 0.8
            
            forecasts = []
            demand = base_demand * factor
            
            for d in range(days):
                seasonal = 1 + 0.15 * np.sin((d / 7) * np.pi * 2)
                noise = 0.9 + np.random.random() * 0.2
                demand *= (1 + growth_rate / 7)
                forecasts.append(int(demand * seasonal * noise))
            
            regions.append({
                "name": name,
                "total": sum(forecasts),
                "peak": max(forecasts),
                "average": int(sum(forecasts) / len(forecasts)),
                "trend": "Growing" if growth_rate > 0 else "Declining" if growth_rate < 0 else "Stable",
                "forecasts": forecasts
            })
        
        total_demand = sum(r["total"] for r in regions)
        peak_demand = max(r["peak"] for r in regions)
        avg_demand = int(total_demand / days)
        
        return jsonify({
            "success": True,
            "summary": {
                "total_demand": total_demand,
                "peak_demand": peak_demand,
                "average_daily": avg_demand,
                "days_forecasted": days,
                "num_regions": num_regions
            },
            "regions": regions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/optimize-distribution", methods=["POST"])
def optimize_distribution():
    """
    Optimize vaccine distribution across locations.
    
    Expected JSON payload:
    {
        "total_doses": 1000000,
        "num_locations": 8,
        "equity_weight": 0.5
    }
    """
    try:
        data = request.get_json()
        
        total_doses = int(data.get("total_doses", 1000000))
        num_locations = int(data.get("num_locations", 8))
        equity_weight = float(data.get("equity_weight", 0.5))
        
        # Sample location data
        location_names = [
            "Metro A", "Metro B", "Suburban C", "Rural D", 
            "Coastal E", "Mountain F", "Valley G", "Urban H",
            "Industrial I", "Academic J"
        ][:num_locations]
        
        locations = []
        total_allocated = 0
        
        for i, name in enumerate(location_names):
            # Generate realistic population and coverage data
            population = int(50000 + np.random.random() * 450000)
            current_coverage = 0.2 + np.random.random() * 0.5
            priority = 0.5 + np.random.random() * 0.5
            
            # Allocation based on need (inverse of coverage) and priority
            need_factor = (1 - current_coverage) * priority
            base_allocation = int(total_doses * need_factor / (num_locations * 0.7))
            
            # Apply equity adjustment
            equity_bonus = int((1 - current_coverage) * equity_weight * total_doses / num_locations * 0.3)
            allocation = min(base_allocation + equity_bonus, int(population * 0.8))
            
            locations.append({
                "name": name,
                "population": population,
                "current_coverage": round(current_coverage * 100, 1),
                "priority": round(priority, 2),
                "allocated_doses": allocation,
                "new_coverage": round((current_coverage + allocation / population) * 100, 1),
                "efficiency_score": round(0.7 + np.random.random() * 0.3, 2)
            })
            total_allocated += allocation
        
        # Normalize allocations to total
        scale = total_doses / total_allocated if total_allocated > 0 else 1
        for loc in locations:
            loc["allocated_doses"] = int(loc["allocated_doses"] * scale)
        
        total_allocated = sum(loc["allocated_doses"] for loc in locations)
        avg_coverage_improvement = sum(loc["new_coverage"] - loc["current_coverage"] for loc in locations) / len(locations)
        
        return jsonify({
            "success": True,
            "summary": {
                "total_distributed": total_allocated,
                "num_locations": num_locations,
                "equity_weight": equity_weight,
                "average_coverage_improvement": round(avg_coverage_improvement, 1)
            },
            "locations": sorted(locations, key=lambda x: -x["allocated_doses"]),
            "recommendations": [
                f"Prioritize {locations[0]['name']} with highest allocation",
                f"Focus on locations below {30 + int(equity_weight * 20)}% coverage for equity goals",
                "Consider cold chain logistics for rural locations"
            ]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    """
    Run the complete VaxFlow pipeline.
    
    Expected JSON payload:
    {
        "production": { ... production params ... },
        "demand": { ... demand params ... },
        "distribution": { ... distribution params ... }
    }
    """
    try:
        data = request.get_json()
        
        # Run production prediction
        prod_data = data.get("production", {})
        production_result = predict_yield_internal(prod_data)
        
        # Run demand forecast
        demand_data = data.get("demand", {})
        demand_result = forecast_demand_internal(demand_data)
        
        # Run distribution optimization
        dist_data = data.get("distribution", {})
        dist_data["total_doses"] = production_result.get("yield", 1000000)
        distribution_result = optimize_distribution_internal(dist_data)
        
        return jsonify({
            "success": True,
            "pipeline_results": {
                "production": production_result,
                "demand": demand_result,
                "distribution": distribution_result
            },
            "summary": {
                "predicted_yield": production_result.get("yield"),
                "total_demand": demand_result.get("total_demand"),
                "doses_distributed": distribution_result.get("total_distributed"),
                "supply_demand_ratio": round(
                    production_result.get("yield", 0) / max(1, demand_result.get("total_demand", 1)),
                    2
                )
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def predict_yield_internal(data):
    """Internal helper for pipeline."""
    params = BatchParameters(
        temperature=float(data.get("temperature", 37.0)),
        humidity=float(data.get("humidity", 55)),
        incubation_time=float(data.get("incubation_time", 72)),
        cell_density=float(data.get("cell_density", 2.5)),
        nutrient_concentration=float(data.get("nutrient_concentration", 15.0)),
        ph_level=float(data.get("ph_level", 7.2)),
        dissolved_oxygen=float(data.get("dissolved_oxygen", 40)),
        batch_size_liters=float(data.get("batch_size", 5000))
    )
    model = get_production_model()
    predicted_yield, _ = model.predict_single(params)
    base_yield = params.batch_size_liters * 500
    efficiency = min(1.0, predicted_yield / base_yield) if base_yield > 0 else 0.5
    return {
        "yield": int(predicted_yield),
        "efficiency": round(efficiency * 100, 1)
    }


def forecast_demand_internal(data):
    """Internal helper for pipeline."""
    days = int(data.get("days", 30))
    num_regions = int(data.get("regions", 5))
    base_demand = int(data.get("base_demand", 50000))
    growth_rate = float(data.get("growth_rate", 5.0)) / 100
    
    total_demand = 0
    for i in range(num_regions):
        factor = 0.6 + np.random.random() * 0.8
        demand = base_demand * factor
        for d in range(days):
            demand *= (1 + growth_rate / 7)
        total_demand += int(demand * days * 0.7)
    
    return {"total_demand": total_demand, "days": days, "regions": num_regions}


def optimize_distribution_internal(data):
    """Internal helper for pipeline."""
    total_doses = int(data.get("total_doses", 1000000))
    return {"total_distributed": total_doses, "locations": int(data.get("num_locations", 8))}


if __name__ == "__main__":
    print("Starting VaxFlow API server on port 5002...")
    app.run(host="0.0.0.0", port=5002, debug=True)
