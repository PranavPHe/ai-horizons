"""
VaxFlow Interactive Analysis Module

Handles user input processing and generates optimization recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FacilityInput:
    """User input for production facility."""
    facility_id: str
    name: str
    bioreactor_capacity_liters: float
    current_inventory: float
    latitude: float
    longitude: float
    cold_chain_capable: bool = True


@dataclass
class EnvironmentInput:
    """User input for environmental/production parameters."""
    temperature: float  # °C
    ph_level: float
    nutrient_concentration: float  # g/L
    humidity: float = 85.0  # %
    dissolved_oxygen: float = 45.0  # %
    incubation_time: float = 72.0  # hours
    cell_density: float = 2.5  # x10^6 cells/mL


@dataclass
class ShippingLocation:
    """User input for shipping destination."""
    location_id: str
    name: str
    latitude: float
    longitude: float
    population: int
    priority: float = 1.0
    current_coverage: float = 0.0
    required_doses: Optional[float] = None


@dataclass
class OptimizationResult:
    """Complete optimization result with recommendations."""
    # Production Analysis
    predicted_yield: float
    yield_efficiency: float  # as percentage of theoretical max
    
    # Recommended Adjustments
    recommended_temperature: float
    temperature_adjustment: float
    recommended_nutrients: float
    nutrient_adjustment: float
    recommended_ph: float
    ph_adjustment: float
    
    # Additional recommendations
    other_recommendations: List[str]
    
    # Distribution Analysis
    distribution_routes: List[Dict[str, Any]]
    total_distance_km: float
    estimated_delivery_time_hours: float
    total_distribution_cost: float
    coverage_by_location: Dict[str, float]
    
    # Warnings/Alerts
    warnings: List[str]
    
    def summary(self) -> str:
        """Generate a readable summary."""
        lines = [
            "=" * 60,
            "VAXFLOW OPTIMIZATION ANALYSIS REPORT",
            "=" * 60,
            "",
            "📊 PRODUCTION ANALYSIS",
            "-" * 40,
            f"  Predicted Yield: {self.predicted_yield:,.0f} doses",
            f"  Yield Efficiency: {self.yield_efficiency:.1f}%",
            "",
            "🔧 RECOMMENDED ADJUSTMENTS",
            "-" * 40,
            f"  Temperature: {self.recommended_temperature:.1f}°C "
            f"({self._format_delta(self.temperature_adjustment)}°C)",
            f"  Nutrient Concentration: {self.recommended_nutrients:.1f} g/L "
            f"({self._format_delta(self.nutrient_adjustment)} g/L)",
            f"  pH Level: {self.recommended_ph:.2f} "
            f"({self._format_delta(self.ph_adjustment)})",
            "",
        ]
        
        if self.other_recommendations:
            lines.append("📋 ADDITIONAL RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(self.other_recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.extend([
            "🚚 DISTRIBUTION OPTIMIZATION",
            "-" * 40,
            f"  Total Routes: {len(self.distribution_routes)}",
            f"  Total Distance: {self.total_distance_km:,.0f} km",
            f"  Est. Delivery Time: {self.estimated_delivery_time_hours:.1f} hours",
            f"  Total Cost: ${self.total_distribution_cost:,.2f}",
            "",
            "📍 Coverage by Location:",
        ])
        
        for loc, coverage in self.coverage_by_location.items():
            lines.append(f"    {loc}: {coverage*100:.1f}%")
        
        if self.warnings:
            lines.extend(["", "⚠️  WARNINGS", "-" * 40])
            for warn in self.warnings:
                lines.append(f"  • {warn}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_delta(self, delta: float) -> str:
        if delta >= 0:
            return f"+{delta:.2f}"
        return f"{delta:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for export."""
        return {
            "report_type": "VaxFlow Optimization Report",
            "generated_at": pd.Timestamp.now().isoformat(),
            "production_analysis": {
                "predicted_yield_doses": self.predicted_yield,
                "yield_efficiency_percent": self.yield_efficiency,
            },
            "production_recommendations": {
                "temperature": {
                    "recommended_value_celsius": self.recommended_temperature,
                    "adjustment_celsius": self.temperature_adjustment
                },
                "nutrient_concentration": {
                    "recommended_value_g_per_L": self.recommended_nutrients,
                    "adjustment_g_per_L": self.nutrient_adjustment
                },
                "ph_level": {
                    "recommended_value": self.recommended_ph,
                    "adjustment": self.ph_adjustment
                },
                "additional_recommendations": self.other_recommendations
            },
            "distribution_optimization": {
                "total_routes": len(self.distribution_routes),
                "total_distance_km": self.total_distance_km,
                "estimated_delivery_hours": self.estimated_delivery_time_hours,
                "total_cost_usd": self.total_distribution_cost,
                "coverage_by_location": self.coverage_by_location
            },
            "shipping_routes": self.distribution_routes,
            "warnings": self.warnings
        }
    
    def export_json(self, filepath: str) -> str:
        """Export full report to JSON file."""
        import json
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    
    def export_csv(self, filepath: str) -> str:
        """Export shipping routes to CSV file."""
        if not self.distribution_routes:
            raise ValueError("No routes to export")
        
        df = pd.DataFrame(self.distribution_routes)
        df.to_csv(filepath, index=False)
        return filepath
    
    def export_report(self, filepath: str) -> str:
        """Export formatted logistics report to text file."""
        from datetime import datetime
        
        lines = [
            "=" * 70,
            "VAXFLOW LOGISTICS REPORT".center(70),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70),
            "=" * 70,
            "",
            "SECTION 1: PRODUCTION EFFICIENCY RECOMMENDATIONS",
            "-" * 70,
            "",
            f"  Predicted Yield:     {self.predicted_yield:>15,.0f} doses",
            f"  Yield Efficiency:    {self.yield_efficiency:>15.1f} %",
            "",
            "  Parameter Adjustments Required:",
            f"    • Temperature:     Adjust to {self.recommended_temperature:.1f}°C "
            f"(change: {self._format_delta(self.temperature_adjustment)}°C)",
            f"    • Nutrients:       Adjust to {self.recommended_nutrients:.1f} g/L "
            f"(change: {self._format_delta(self.nutrient_adjustment)} g/L)",
            f"    • pH Level:        Adjust to {self.recommended_ph:.2f} "
            f"(change: {self._format_delta(self.ph_adjustment)})",
            "",
        ]
        
        if self.other_recommendations:
            lines.append("  Additional Recommendations:")
            for i, rec in enumerate(self.other_recommendations, 1):
                # Wrap long recommendations
                lines.append(f"    {i}. {rec}")
            lines.append("")
        
        lines.extend([
            "",
            "SECTION 2: OPTIMIZED SHIPPING ROUTES",
            "-" * 70,
            "",
            f"  Total Routes:        {len(self.distribution_routes):>15}",
            f"  Total Distance:      {self.total_distance_km:>15,.0f} km",
            f"  Max Delivery Time:   {self.estimated_delivery_time_hours:>15.1f} hours",
            f"  Total Shipping Cost: ${self.total_distribution_cost:>14,.2f}",
            "",
            "  Route Details:",
            "  " + "-" * 66,
            f"  {'From':<20} {'To':<20} {'Doses':>12} {'Dist(km)':>10} {'Cost($)':>10}",
            "  " + "-" * 66,
        ])
        
        for route in self.distribution_routes:
            from_name = route.get('from_name', route.get('from_facility', 'N/A'))[:18]
            to_name = route.get('to_name', route.get('to_location', 'N/A'))[:18]
            lines.append(
                f"  {from_name:<20} {to_name:<20} "
                f"{route['doses']:>12,.0f} {route['distance_km']:>10.0f} "
                f"{route['cost']:>10.2f}"
            )
        
        lines.extend([
            "  " + "-" * 66,
            "",
            "SECTION 3: COVERAGE ANALYSIS",
            "-" * 70,
            "",
            f"  {'Location':<30} {'Coverage':>15} {'Status':<20}",
            "  " + "-" * 66,
        ])
        
        for loc, coverage in self.coverage_by_location.items():
            status = "✓ ADEQUATE" if coverage >= 0.9 else ("⚠ PARTIAL" if coverage >= 0.5 else "✗ CRITICAL")
            lines.append(f"  {loc:<30} {coverage*100:>14.1f}% {status:<20}")
        
        lines.append("  " + "-" * 66)
        
        if self.warnings:
            lines.extend([
                "",
                "SECTION 4: ALERTS & WARNINGS",
                "-" * 70,
            ])
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")
        
        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT".center(70),
            "=" * 70,
        ])
        
        report_text = "\n".join(lines)
        
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        return filepath
    
    def export_all(self, base_path: str = "vaxflow_report") -> Dict[str, str]:
        """
        Export all report formats.
        
        Args:
            base_path: Base filename (without extension)
        
        Returns:
            Dict with paths to exported files
        """
        exports = {}
        
        exports['json'] = self.export_json(f"{base_path}.json")
        exports['csv'] = self.export_csv(f"{base_path}_routes.csv")
        exports['txt'] = self.export_report(f"{base_path}.txt")
        
        return exports


class VaxFlowAnalyzer:
    """
    Interactive analyzer that processes user inputs and provides
    production optimization and distribution recommendations.
    
    Example usage:
    ```python
    analyzer = VaxFlowAnalyzer()
    
    # Add facilities
    analyzer.add_facility(FacilityInput(
        facility_id="F1",
        name="Main Plant",
        bioreactor_capacity_liters=5000,
        current_inventory=100000,
        latitude=40.7,
        longitude=-74.0
    ))
    
    # Set current environmental parameters
    analyzer.set_environment(EnvironmentInput(
        temperature=36.5,
        ph_level=7.1,
        nutrient_concentration=12.0
    ))
    
    # Add shipping locations
    analyzer.add_shipping_location(ShippingLocation(
        location_id="NYC",
        name="New York City",
        latitude=40.7,
        longitude=-74.0,
        population=8_000_000
    ))
    
    # Run analysis
    result = analyzer.analyze()
    print(result.summary())
    ```
    """
    
    # Optimal values based on training data patterns
    OPTIMAL_TEMPERATURE = 37.0
    OPTIMAL_PH = 7.2
    OPTIMAL_NUTRIENTS = 15.0
    OPTIMAL_HUMIDITY = 85.0
    OPTIMAL_OXYGEN = 45.0
    OPTIMAL_TIME = 72.0
    
    # Yield model coefficients (from production_predictor training)
    BASE_YIELD_PER_LITER = 1000  # doses per liter at optimal conditions
    
    def __init__(self):
        self.facilities: Dict[str, FacilityInput] = {}
        self.environment: Optional[EnvironmentInput] = None
        self.shipping_locations: Dict[str, ShippingLocation] = {}
        self._predictor = None
        self._optimizer = None
    
    def add_facility(self, facility: FacilityInput) -> "VaxFlowAnalyzer":
        """Add a production facility."""
        self.facilities[facility.facility_id] = facility
        logger.info(f"Added facility: {facility.name} ({facility.facility_id})")
        return self
    
    def set_environment(self, env: EnvironmentInput) -> "VaxFlowAnalyzer":
        """Set current environmental/production parameters."""
        self.environment = env
        logger.info(f"Set environment: temp={env.temperature}°C, pH={env.ph_level}")
        return self
    
    def add_shipping_location(self, location: ShippingLocation) -> "VaxFlowAnalyzer":
        """Add a shipping destination."""
        self.shipping_locations[location.location_id] = location
        logger.info(f"Added shipping location: {location.name}")
        return self
    
    def clear(self) -> "VaxFlowAnalyzer":
        """Clear all inputs."""
        self.facilities.clear()
        self.environment = None
        self.shipping_locations.clear()
        return self
    
    def _calculate_yield(self, env: EnvironmentInput, capacity_liters: float) -> float:
        """
        Calculate predicted yield based on environmental parameters.
        Uses a polynomial model based on deviations from optimal.
        """
        # Calculate efficiency factors for each parameter
        temp_eff = 1.0 - 0.1 * ((env.temperature - self.OPTIMAL_TEMPERATURE) ** 2)
        ph_eff = 1.0 - 0.5 * ((env.ph_level - self.OPTIMAL_PH) ** 2)
        nutrient_eff = 1.0 - 0.02 * ((env.nutrient_concentration - self.OPTIMAL_NUTRIENTS) ** 2)
        humidity_eff = 1.0 - 0.005 * ((env.humidity - self.OPTIMAL_HUMIDITY) ** 2)
        oxygen_eff = 1.0 - 0.002 * ((env.dissolved_oxygen - self.OPTIMAL_OXYGEN) ** 2)
        time_eff = 1.0 - 0.001 * ((env.incubation_time - self.OPTIMAL_TIME) ** 2)
        
        # Cell density multiplier
        density_mult = 0.5 + 0.2 * env.cell_density
        
        # Combined efficiency (clipped to reasonable range)
        total_eff = np.clip(
            temp_eff * ph_eff * nutrient_eff * humidity_eff * oxygen_eff * time_eff * density_mult,
            0.1, 1.5
        )
        
        # Calculate yield
        base_yield = capacity_liters * self.BASE_YIELD_PER_LITER
        return base_yield * total_eff
    
    def _calculate_optimal_adjustments(
        self, 
        env: EnvironmentInput
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate recommended parameter adjustments."""
        # Calculate how far off from optimal and recommend adjustments
        temp_adj = self.OPTIMAL_TEMPERATURE - env.temperature
        ph_adj = self.OPTIMAL_PH - env.ph_level
        nutrient_adj = self.OPTIMAL_NUTRIENTS - env.nutrient_concentration
        
        # Limit adjustments to safe ranges
        temp_adj = np.clip(temp_adj, -2.0, 2.0)  # Max 2°C change
        ph_adj = np.clip(ph_adj, -0.3, 0.3)  # Max 0.3 pH change
        nutrient_adj = np.clip(nutrient_adj, -5.0, 5.0)  # Max 5 g/L change
        
        rec_temp = env.temperature + temp_adj
        rec_ph = env.ph_level + ph_adj
        rec_nutrients = env.nutrient_concentration + nutrient_adj
        
        return rec_temp, temp_adj, rec_nutrients, nutrient_adj, rec_ph, ph_adj
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between facilities and shipping locations."""
        if not self.facilities or not self.shipping_locations:
            return np.array([])
        
        facility_coords = np.array([
            [f.latitude, f.longitude] for f in self.facilities.values()
        ])
        location_coords = np.array([
            [l.latitude, l.longitude] for l in self.shipping_locations.values()
        ])
        
        # Haversine approximation (degrees to km)
        # Using simple Euclidean * 111 km/degree
        distances = np.zeros((len(self.facilities), len(self.shipping_locations)))
        for i, fc in enumerate(facility_coords):
            for j, lc in enumerate(location_coords):
                lat_diff = (fc[0] - lc[0]) * 111
                lon_diff = (fc[1] - lc[1]) * 111 * np.cos(np.radians(fc[0]))
                distances[i, j] = np.sqrt(lat_diff**2 + lon_diff**2)
        
        return distances
    
    def _optimize_distribution(
        self, 
        total_supply: float
    ) -> Tuple[List[Dict], float, float, float, Dict[str, float]]:
        """Optimize distribution routes to minimize distance and delays."""
        if not self.facilities or not self.shipping_locations:
            return [], 0, 0, 0, {}
        
        distances = self._calculate_distances()
        facility_ids = list(self.facilities.keys())
        location_ids = list(self.shipping_locations.keys())
        
        # Calculate demand per location
        demands = {}
        for loc_id, loc in self.shipping_locations.items():
            if loc.required_doses is not None:
                demands[loc_id] = loc.required_doses
            else:
                # Default: proportional to population * priority
                demands[loc_id] = loc.population * 0.05 * loc.priority
        
        total_demand = sum(demands.values())
        
        # Calculate inventory per facility
        inventories = {f_id: f.current_inventory for f_id, f in self.facilities.items()}
        
        # Greedy allocation: assign from nearest facility with inventory
        routes = []
        allocated = {loc_id: 0.0 for loc_id in location_ids}
        remaining_inv = inventories.copy()
        
        # Sort locations by priority (descending)
        sorted_locs = sorted(
            location_ids,
            key=lambda x: self.shipping_locations[x].priority,
            reverse=True
        )
        
        for loc_idx, loc_id in enumerate(sorted_locs):
            loc = self.shipping_locations[loc_id]
            needed = demands[loc_id] - allocated[loc_id]
            
            if needed <= 0:
                continue
            
            # Find nearest facility with inventory
            j = location_ids.index(loc_id)
            facility_dists = [(f_id, distances[i, j]) for i, f_id in enumerate(facility_ids)]
            facility_dists.sort(key=lambda x: x[1])
            
            for f_id, dist in facility_dists:
                if remaining_inv[f_id] <= 0:
                    continue
                
                # Allocate from this facility
                alloc = min(needed, remaining_inv[f_id])
                if alloc > 0:
                    routes.append({
                        "from_facility": f_id,
                        "from_name": self.facilities[f_id].name,
                        "to_location": loc_id,
                        "to_name": loc.name,
                        "doses": alloc,
                        "distance_km": dist,
                        "time_hours": dist / 60,  # Assume 60 km/h average
                        "cost": dist * 0.5 * (alloc / 1000)  # $0.5/km per 1000 doses
                    })
                    
                    allocated[loc_id] += alloc
                    remaining_inv[f_id] -= alloc
                    needed -= alloc
                
                if needed <= 0:
                    break
        
        # Calculate totals
        total_distance = sum(r["distance_km"] for r in routes)
        max_time = max((r["time_hours"] for r in routes), default=0)
        total_cost = sum(r["cost"] for r in routes)
        
        # Calculate coverage
        coverage = {}
        for loc_id in location_ids:
            if demands[loc_id] > 0:
                coverage[loc_id] = min(1.0, allocated[loc_id] / demands[loc_id])
            else:
                coverage[loc_id] = 1.0
        
        return routes, total_distance, max_time, total_cost, coverage
    
    def _generate_recommendations(
        self,
        env: EnvironmentInput,
        yield_efficiency: float
    ) -> List[str]:
        """Generate additional actionable recommendations."""
        recommendations = []
        
        # Temperature recommendations
        if abs(env.temperature - self.OPTIMAL_TEMPERATURE) > 1.0:
            if env.temperature < self.OPTIMAL_TEMPERATURE:
                recommendations.append(
                    "Increase bioreactor temperature gradually (0.5°C/hour) to avoid thermal shock"
                )
            else:
                recommendations.append(
                    "Reduce cooling system setpoint; ensure adequate ventilation"
                )
        
        # pH recommendations
        if abs(env.ph_level - self.OPTIMAL_PH) > 0.2:
            if env.ph_level < self.OPTIMAL_PH:
                recommendations.append(
                    "Add sodium bicarbonate buffer to increase pH; monitor CO2 levels"
                )
            else:
                recommendations.append(
                    "Increase CO2 sparging or add acidic buffer to lower pH"
                )
        
        # Nutrient recommendations
        if env.nutrient_concentration < 12:
            recommendations.append(
                "Supplement with glucose and amino acids to boost nutrient levels"
            )
        elif env.nutrient_concentration > 20:
            recommendations.append(
                "High nutrient levels may cause metabolic overflow; consider dilution"
            )
        
        # Oxygen recommendations
        if env.dissolved_oxygen < 35:
            recommendations.append(
                "Increase agitation speed or oxygen sparging to improve DO levels"
            )
        elif env.dissolved_oxygen > 55:
            recommendations.append(
                "Reduce oxygen flow to prevent oxidative stress on cells"
            )
        
        # Efficiency-based recommendations
        if yield_efficiency < 70:
            recommendations.append(
                "Consider cell line passage optimization; high-passage cells may show reduced productivity"
            )
        
        # Incubation time
        if env.incubation_time < 60:
            recommendations.append(
                "Extend incubation period to allow complete cell growth phase"
            )
        elif env.incubation_time > 84:
            recommendations.append(
                "Monitor for cell viability decline in extended incubation"
            )
        
        return recommendations
    
    def analyze(self) -> OptimizationResult:
        """
        Run complete analysis on provided inputs.
        
        Returns:
            OptimizationResult with predictions and recommendations
        """
        if not self.facilities:
            raise ValueError("No facilities added. Use add_facility() first.")
        if self.environment is None:
            raise ValueError("Environment not set. Use set_environment() first.")
        if not self.shipping_locations:
            raise ValueError("No shipping locations. Use add_shipping_location() first.")
        
        env = self.environment
        warnings = []
        
        # 1. Calculate total capacity and predicted yield
        total_capacity = sum(f.bioreactor_capacity_liters for f in self.facilities.values())
        predicted_yield = self._calculate_yield(env, total_capacity)
        
        # Calculate efficiency vs theoretical maximum
        theoretical_max = total_capacity * self.BASE_YIELD_PER_LITER * 1.5
        yield_efficiency = (predicted_yield / theoretical_max) * 100
        
        # 2. Calculate optimal adjustments
        (rec_temp, temp_adj, rec_nutrients, nutrient_adj, 
         rec_ph, ph_adj) = self._calculate_optimal_adjustments(env)
        
        # 3. Generate additional recommendations
        other_recs = self._generate_recommendations(env, yield_efficiency)
        
        # 4. Optimize distribution
        total_inventory = sum(f.current_inventory for f in self.facilities.values())
        routes, total_dist, delivery_time, total_cost, coverage = self._optimize_distribution(
            total_inventory
        )
        
        # 5. Generate warnings
        if yield_efficiency < 50:
            warnings.append(
                f"Low yield efficiency ({yield_efficiency:.1f}%). "
                "Review all environmental parameters."
            )
        
        if env.temperature < 35 or env.temperature > 39:
            warnings.append(
                f"Temperature {env.temperature}°C outside safe range (35-39°C)"
            )
        
        if env.ph_level < 6.8 or env.ph_level > 7.6:
            warnings.append(
                f"pH {env.ph_level} outside optimal range (6.8-7.6)"
            )
        
        avg_coverage = np.mean(list(coverage.values())) if coverage else 0
        if avg_coverage < 0.8:
            warnings.append(
                f"Average coverage {avg_coverage*100:.1f}% below target. "
                "Consider increasing inventory or production."
            )
        
        # Sort routes by distance (shortest first for efficiency report)
        routes.sort(key=lambda x: x["distance_km"])
        
        return OptimizationResult(
            predicted_yield=predicted_yield,
            yield_efficiency=yield_efficiency,
            recommended_temperature=rec_temp,
            temperature_adjustment=temp_adj,
            recommended_nutrients=rec_nutrients,
            nutrient_adjustment=nutrient_adj,
            recommended_ph=rec_ph,
            ph_adjustment=ph_adj,
            other_recommendations=other_recs,
            distribution_routes=routes,
            total_distance_km=total_dist,
            estimated_delivery_time_hours=delivery_time,
            total_distribution_cost=total_cost,
            coverage_by_location=coverage,
            warnings=warnings
        )


def interactive_session():
    """
    Run an interactive command-line session for user input.
    """
    print("\n" + "=" * 60)
    print("VAXFLOW INTERACTIVE ANALYSIS")
    print("=" * 60)
    print("\nThis tool analyzes your production parameters and optimizes")
    print("vaccine manufacturing conditions and distribution routes.\n")
    
    analyzer = VaxFlowAnalyzer()
    
    # Collect facility data
    print("-" * 40)
    print("STEP 1: PRODUCTION FACILITY DATA")
    print("-" * 40)
    
    while True:
        print("\nEnter facility details (or 'done' to continue):")
        facility_id = input("  Facility ID: ").strip()
        if facility_id.lower() == 'done':
            break
        
        name = input("  Facility Name: ").strip() or f"Facility {facility_id}"
        capacity = float(input("  Bioreactor Capacity (liters): ") or 1000)
        inventory = float(input("  Current Inventory (doses): ") or 100000)
        lat = float(input("  Latitude: ") or 40.0)
        lon = float(input("  Longitude: ") or -74.0)
        
        analyzer.add_facility(FacilityInput(
            facility_id=facility_id,
            name=name,
            bioreactor_capacity_liters=capacity,
            current_inventory=inventory,
            latitude=lat,
            longitude=lon
        ))
        print(f"  ✓ Added facility: {name}")
    
    if not analyzer.facilities:
        # Add default facility
        print("\n  No facilities entered. Using default facility.")
        analyzer.add_facility(FacilityInput(
            facility_id="F1",
            name="Default Plant",
            bioreactor_capacity_liters=5000,
            current_inventory=500000,
            latitude=40.7,
            longitude=-74.0
        ))
    
    # Collect environmental parameters
    print("\n" + "-" * 40)
    print("STEP 2: ENVIRONMENTAL PARAMETERS")
    print("-" * 40)
    
    print("\nEnter current production parameters:")
    temp = float(input("  Temperature (°C) [default: 37]: ") or 37)
    ph = float(input("  pH Level [default: 7.2]: ") or 7.2)
    nutrients = float(input("  Nutrient Concentration (g/L) [default: 15]: ") or 15)
    humidity = float(input("  Humidity (%) [default: 85]: ") or 85)
    oxygen = float(input("  Dissolved Oxygen (%) [default: 45]: ") or 45)
    incubation = float(input("  Incubation Time (hours) [default: 72]: ") or 72)
    density = float(input("  Cell Density (x10^6/mL) [default: 2.5]: ") or 2.5)
    
    analyzer.set_environment(EnvironmentInput(
        temperature=temp,
        ph_level=ph,
        nutrient_concentration=nutrients,
        humidity=humidity,
        dissolved_oxygen=oxygen,
        incubation_time=incubation,
        cell_density=density
    ))
    print("  ✓ Environment parameters set")
    
    # Collect shipping locations
    print("\n" + "-" * 40)
    print("STEP 3: SHIPPING LOCATIONS")
    print("-" * 40)
    
    while True:
        print("\nEnter shipping location (or 'done' to continue):")
        loc_id = input("  Location ID: ").strip()
        if loc_id.lower() == 'done':
            break
        
        name = input("  Location Name: ").strip() or f"Location {loc_id}"
        lat = float(input("  Latitude: ") or 35.0)
        lon = float(input("  Longitude: ") or -80.0)
        pop = int(input("  Population: ") or 1000000)
        priority = float(input("  Priority (0.5-2.0) [default: 1.0]: ") or 1.0)
        doses_input = input("  Required Doses (or blank for auto): ").strip()
        required = float(doses_input) if doses_input else None
        
        analyzer.add_shipping_location(ShippingLocation(
            location_id=loc_id,
            name=name,
            latitude=lat,
            longitude=lon,
            population=pop,
            priority=priority,
            required_doses=required
        ))
        print(f"  ✓ Added location: {name}")
    
    if not analyzer.shipping_locations:
        # Add default locations
        print("\n  No locations entered. Using default locations.")
        for loc in [
            ("NYC", "New York City", 40.7, -74.0, 8_000_000),
            ("LA", "Los Angeles", 34.1, -118.2, 4_000_000),
            ("CHI", "Chicago", 41.9, -87.6, 2_700_000),
        ]:
            analyzer.add_shipping_location(ShippingLocation(
                location_id=loc[0],
                name=loc[1],
                latitude=loc[2],
                longitude=loc[3],
                population=loc[4]
            ))
    
    # Run analysis
    print("\n" + "-" * 40)
    print("ANALYZING...")
    print("-" * 40)
    
    result = analyzer.analyze()
    print("\n" + result.summary())
    
    # Export reports
    print("\n" + "-" * 40)
    print("EXPORTING REPORTS...")
    print("-" * 40)
    
    export_choice = input("\nExport logistics reports? (y/n) [default: y]: ").strip().lower()
    if export_choice != 'n':
        base_name = input("Report filename prefix [default: vaxflow_report]: ").strip() or "vaxflow_report"
        exports = result.export_all(base_name)
        print(f"\n  ✓ JSON Report:     {exports['json']}")
        print(f"  ✓ Routes CSV:      {exports['csv']}")
        print(f"  ✓ Full Report:     {exports['txt']}")
        print("\n  Reports exported successfully!")
    
    return result


def analyze_from_data(
    facilities_df: pd.DataFrame,
    environment: Dict[str, float],
    locations_df: pd.DataFrame
) -> OptimizationResult:
    """
    Analyze from pandas DataFrames.
    
    Args:
        facilities_df: DataFrame with columns: id, name, capacity, inventory, latitude, longitude
        environment: Dict with keys: temperature, ph_level, nutrient_concentration, etc.
        locations_df: DataFrame with columns: id, name, latitude, longitude, population, priority
    
    Returns:
        OptimizationResult
    """
    analyzer = VaxFlowAnalyzer()
    
    # Add facilities
    for _, row in facilities_df.iterrows():
        analyzer.add_facility(FacilityInput(
            facility_id=str(row.get("id", row.name)),
            name=str(row.get("name", "")),
            bioreactor_capacity_liters=row.get("capacity", row.get("bioreactor_capacity", 1000)),
            current_inventory=row.get("inventory", row.get("current_inventory", 0)),
            latitude=row["latitude"],
            longitude=row["longitude"],
            cold_chain_capable=row.get("cold_chain", True)
        ))
    
    # Set environment
    analyzer.set_environment(EnvironmentInput(
        temperature=environment.get("temperature", 37.0),
        ph_level=environment.get("ph_level", environment.get("ph", 7.2)),
        nutrient_concentration=environment.get("nutrient_concentration", environment.get("nutrients", 15.0)),
        humidity=environment.get("humidity", 85.0),
        dissolved_oxygen=environment.get("dissolved_oxygen", 45.0),
        incubation_time=environment.get("incubation_time", 72.0),
        cell_density=environment.get("cell_density", 2.5)
    ))
    
    # Add locations
    for _, row in locations_df.iterrows():
        analyzer.add_shipping_location(ShippingLocation(
            location_id=str(row.get("id", row.name)),
            name=str(row.get("name", "")),
            latitude=row["latitude"],
            longitude=row["longitude"],
            population=int(row.get("population", 100000)),
            priority=row.get("priority", 1.0),
            current_coverage=row.get("coverage", 0.0),
            required_doses=row.get("required_doses", None)
        ))
    
    return analyzer.analyze()


if __name__ == "__main__":
    interactive_session()
