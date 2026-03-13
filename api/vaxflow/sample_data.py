"""
Sample Data Generators for VaxFlow

Generates synthetic data for testing and demonstration purposes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd


def generate_production_data(
    n_batches: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic vaccine production batch data.
    
    Args:
        n_batches: Number of batches to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with production parameters and yields
    """
    rng = np.random.default_rng(random_state)
    
    # Generate realistic parameter ranges
    data = {
        "batch_id": [f"BATCH_{i:05d}" for i in range(n_batches)],
        "temperature": rng.normal(37.0, 0.5, n_batches).clip(35, 39),
        "humidity": rng.normal(85, 5, n_batches).clip(70, 95),
        "incubation_time": rng.normal(72, 8, n_batches).clip(48, 96),
        "cell_density": rng.normal(2.5, 0.5, n_batches).clip(1, 5),
        "nutrient_concentration": rng.normal(15, 2, n_batches).clip(10, 25),
        "ph_level": rng.normal(7.2, 0.15, n_batches).clip(6.8, 7.6),
        "dissolved_oxygen": rng.normal(45, 5, n_batches).clip(30, 60),
    }
    
    df = pd.DataFrame(data)
    
    # Generate yield based on realistic relationships
    # Optimal: temp=37, humidity=85, time=72, cell_density=2.5, nutrients=15, pH=7.2, DO=45
    base_yield = 1e6  # 1 million doses baseline
    
    temp_effect = -50000 * (df["temperature"] - 37) ** 2
    humidity_effect = -1000 * (df["humidity"] - 85) ** 2
    time_effect = 5000 * df["incubation_time"] - 100 * (df["incubation_time"] - 72) ** 2
    density_effect = 100000 * df["cell_density"]
    nutrient_effect = 10000 * df["nutrient_concentration"]
    ph_effect = -500000 * (df["ph_level"] - 7.2) ** 2
    do_effect = 5000 * df["dissolved_oxygen"] - 50 * (df["dissolved_oxygen"] - 45) ** 2
    
    noise = rng.normal(0, 50000, n_batches)
    
    df["yield"] = (
        base_yield + temp_effect + humidity_effect + time_effect +
        density_effect + nutrient_effect + ph_effect + do_effect + noise
    ).clip(0)
    
    # Add some metadata
    df["date"] = [
        datetime(2025, 1, 1) + timedelta(days=i // 5)
        for i in range(n_batches)
    ]
    df["facility"] = rng.choice(["F1", "F2", "F3"], n_batches)
    df["cell_line"] = rng.choice(["HEK293", "CHO", "Vero"], n_batches, p=[0.5, 0.3, 0.2])
    
    return df


def generate_demand_history(
    n_days: int = 365,
    n_regions: int = 5,
    start_date: datetime = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic vaccine demand history.
    
    Args:
        n_days: Number of days of history
        n_regions: Number of regions
        start_date: Start date for history
        random_state: Random seed
    
    Returns:
        DataFrame with daily demand by region
    """
    rng = np.random.default_rng(random_state)
    
    if start_date is None:
        start_date = datetime(2025, 1, 1)
    
    region_names = ["North", "South", "East", "West", "Central"][:n_regions]
    region_base_demand = {
        "North": 50000,
        "South": 80000,
        "East": 60000,
        "West": 70000,
        "Central": 90000
    }
    
    records = []
    for day_idx in range(n_days):
        date = start_date + timedelta(days=day_idx)
        
        # Add weekly seasonality (higher on weekdays)
        dow_factor = 1.2 if date.weekday() < 5 else 0.7
        
        # Add monthly trend (vaccine campaigns)
        month_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
        
        # Add overall trend (vaccination rate decreases over time)
        trend_factor = 1.5 - 0.5 * (day_idx / n_days)
        
        for region in region_names:
            base = region_base_demand.get(region, 50000)
            
            # Random variation
            noise = rng.normal(0, base * 0.15)
            
            demand = base * dow_factor * month_factor * trend_factor + noise
            demand = max(0, demand)
            
            records.append({
                "date": date,
                "region": region,
                "demand": demand
            })
    
    return pd.DataFrame(records)


def generate_facility_data(
    n_facilities: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic facility data.
    
    Args:
        n_facilities: Number of facilities
        random_state: Random seed
    
    Returns:
        DataFrame with facility information
    """
    rng = np.random.default_rng(random_state)
    
    # US-centric coordinates for demonstration
    facilities = [
        {"id": "F1", "name": "Northeast Plant", "latitude": 41.5, "longitude": -72.8},
        {"id": "F2", "name": "Southeast Center", "latitude": 33.5, "longitude": -84.3},
        {"id": "F3", "name": "Midwest Hub", "latitude": 41.8, "longitude": -87.6},
        {"id": "F4", "name": "Southwest Facility", "latitude": 33.4, "longitude": -112.0},
        {"id": "F5", "name": "West Coast Plant", "latitude": 37.8, "longitude": -122.4},
    ][:n_facilities]
    
    df = pd.DataFrame(facilities)
    df["capacity"] = rng.uniform(500000, 2000000, n_facilities)
    df["inventory"] = df["capacity"] * rng.uniform(0.2, 0.8, n_facilities)
    df["cold_chain"] = True
    df["cost_per_dose"] = rng.uniform(0.5, 1.5, n_facilities)
    
    return df


def generate_region_data(
    n_regions: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic region data.
    
    Args:
        n_regions: Number of regions
        random_state: Random seed
    
    Returns:
        DataFrame with region information
    """
    rng = np.random.default_rng(random_state)
    
    regions = [
        {"id": "R_North", "name": "North", "latitude": 44.0, "longitude": -93.0, "population": 15000000},
        {"id": "R_South", "name": "South", "latitude": 32.0, "longitude": -90.0, "population": 25000000},
        {"id": "R_East", "name": "East", "latitude": 40.0, "longitude": -75.0, "population": 30000000},
        {"id": "R_West", "name": "West", "latitude": 36.0, "longitude": -118.0, "population": 20000000},
        {"id": "R_Central", "name": "Central", "latitude": 39.0, "longitude": -98.0, "population": 18000000},
    ][:n_regions]
    
    df = pd.DataFrame(regions)
    df["priority"] = rng.uniform(0.8, 1.2, n_regions)
    df["coverage"] = rng.uniform(0.1, 0.4, n_regions)  # Current vaccination coverage
    df["min_allocation"] = df["population"] * 0.01  # Minimum 1% allocation
    
    return df


def generate_complete_dataset(
    n_production_batches: int = 500,
    n_demand_days: int = 365,
    n_facilities: int = 5,
    n_regions: int = 5,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate a complete dataset for VaxFlow testing.
    
    Returns:
        Dict with 'production', 'demand', 'facilities', 'regions' DataFrames
    """
    return {
        "production": generate_production_data(n_production_batches, random_state),
        "demand": generate_demand_history(n_demand_days, n_regions, random_state=random_state),
        "facilities": generate_facility_data(n_facilities, random_state),
        "regions": generate_region_data(n_regions, random_state)
    }


def save_sample_data(output_dir: str = "./data"):
    """Save sample datasets to files."""
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets = generate_complete_dataset()
    
    for name, df in datasets.items():
        filepath = Path(output_dir) / f"{name}_data.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath}")
    
    return datasets


if __name__ == "__main__":
    # Generate and display sample data
    data = generate_complete_dataset()
    
    print("Production Data Sample:")
    print(data["production"].head())
    print(f"\nShape: {data['production'].shape}")
    
    print("\n" + "="*50)
    print("Demand History Sample:")
    print(data["demand"].head(10))
    print(f"\nShape: {data['demand'].shape}")
    
    print("\n" + "="*50)
    print("Facilities:")
    print(data["facilities"])
    
    print("\n" + "="*50)
    print("Regions:")
    print(data["regions"])
