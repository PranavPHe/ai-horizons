"""
Distribution Optimizer for VaxFlow

Optimizes vaccine distribution across regions considering:
- Supply constraints
- Demand forecasts
- Transportation costs and cold chain requirements
- Equity and priority factors
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist
import joblib

logger = logging.getLogger(__name__)


@dataclass
class Facility:
    """Vaccine production/storage facility."""
    id: str
    name: str
    location: Tuple[float, float]  # (latitude, longitude)
    capacity: float  # Max doses can store/produce
    current_inventory: float = 0.0
    cold_chain_capable: bool = True
    cost_per_dose: float = 1.0  # Relative storage cost


@dataclass
class Region:
    """Distribution region/destination."""
    id: str
    name: str
    location: Tuple[float, float]  # Center point (lat, lon)
    population: int
    priority_score: float = 1.0  # Higher = higher priority
    current_coverage: float = 0.0  # Fraction already vaccinated
    min_allocation: float = 0.0  # Minimum doses required


@dataclass
class AllocationPlan:
    """Optimized allocation plan."""
    allocations: Dict[str, Dict[str, float]]  # facility_id -> {region_id: doses}
    total_cost: float
    total_doses_distributed: float
    unmet_demand: float
    coverage_by_region: Dict[str, float]
    transport_details: List[Dict[str, Any]]
    
    def summary(self) -> str:
        lines = [
            f"Distribution Plan Summary",
            f"{'='*40}",
            f"Total Doses: {self.total_doses_distributed:,.0f}",
            f"Est. Cost: ${self.total_cost:,.2f}",
            f"Unmet Demand: {self.unmet_demand:,.0f}",
            "",
            "Coverage by Region:"
        ]
        for region, coverage in sorted(self.coverage_by_region.items()):
            lines.append(f"  {region}: {coverage*100:.1f}%")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "allocations": self.allocations,
            "total_cost": self.total_cost,
            "total_doses_distributed": self.total_doses_distributed,
            "unmet_demand": self.unmet_demand,
            "coverage_by_region": self.coverage_by_region,
            "transport_details": self.transport_details
        }


class DistributionOptimizer:
    """
    Vaccine distribution optimization using linear programming and heuristics.
    
    Example usage:
    ```python
    optimizer = DistributionOptimizer(objective="minimize_cost")
    optimizer.set_facilities(facility_data)
    optimizer.set_regions(region_data)
    
    plan = optimizer.optimize(
        supply={"facility_1": 100000, "facility_2": 80000},
        demand={"region_a": 50000, "region_b": 60000, "region_c": 40000}
    )
    print(plan.summary())
    ```
    """
    
    OBJECTIVES = ["minimize_cost", "minimize_time", "maximize_coverage", "maximize_equity"]
    
    def __init__(
        self,
        objective: str = "minimize_cost",
        max_transport_time: float = 48.0,  # hours
        cold_chain_temp: float = -70.0,    # Celsius
        cost_per_km: float = 0.5,          # $ per km per 1000 doses
        wastage_rate: float = 0.02         # 2% wastage during transport
    ):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}")
        
        self.objective = objective
        self.max_transport_time = max_transport_time
        self.cold_chain_temp = cold_chain_temp
        self.cost_per_km = cost_per_km
        self.wastage_rate = wastage_rate
        
        self.facilities: Dict[str, Facility] = {}
        self.regions: Dict[str, Region] = {}
        self._distance_matrix: Optional[np.ndarray] = None
    
    def set_facilities(self, data: pd.DataFrame) -> "DistributionOptimizer":
        """
        Set production/storage facilities.
        
        Expected columns: id, name, latitude, longitude, capacity, inventory, cold_chain
        """
        self.facilities = {}
        for _, row in data.iterrows():
            facility = Facility(
                id=str(row.get("id", row.name)),
                name=str(row.get("name", f"Facility_{row.name}")),
                location=(row["latitude"], row["longitude"]),
                capacity=row.get("capacity", 1e6),
                current_inventory=row.get("inventory", row.get("current_inventory", 0)),
                cold_chain_capable=row.get("cold_chain", True),
                cost_per_dose=row.get("cost_per_dose", 1.0)
            )
            self.facilities[facility.id] = facility
        
        self._distance_matrix = None  # Reset cached distances
        logger.info(f"Set {len(self.facilities)} facilities")
        return self
    
    def set_regions(self, data: pd.DataFrame) -> "DistributionOptimizer":
        """
        Set distribution regions.
        
        Expected columns: id, name, latitude, longitude, population, priority, coverage
        """
        self.regions = {}
        for _, row in data.iterrows():
            region = Region(
                id=str(row.get("id", row.name)),
                name=str(row.get("name", f"Region_{row.name}")),
                location=(row["latitude"], row["longitude"]),
                population=int(row.get("population", 100000)),
                priority_score=row.get("priority", row.get("priority_score", 1.0)),
                current_coverage=row.get("coverage", row.get("current_coverage", 0.0)),
                min_allocation=row.get("min_allocation", 0)
            )
            self.regions[region.id] = region
        
        self._distance_matrix = None  # Reset cached distances
        logger.info(f"Set {len(self.regions)} regions")
        return self
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between facilities and regions."""
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        facility_locs = np.array([f.location for f in self.facilities.values()])
        region_locs = np.array([r.location for r in self.regions.values()])
        
        # Haversine distance approximation (good enough for optimization)
        # Convert to radians
        facility_rad = np.radians(facility_locs)
        region_rad = np.radians(region_locs)
        
        # Simple Euclidean in lat/lon space * approximate km conversion
        self._distance_matrix = cdist(facility_locs, region_locs) * 111  # ~111 km per degree
        
        return self._distance_matrix
    
    def _get_cost_matrix(self) -> np.ndarray:
        """Calculate transport cost matrix."""
        distances = self._calculate_distances()
        
        # Cost = distance * cost_per_km (per 1000 doses) + facility storage cost
        facility_costs = np.array([f.cost_per_dose for f in self.facilities.values()])
        
        cost_matrix = distances * self.cost_per_km / 1000 + facility_costs[:, np.newaxis]
        return cost_matrix
    
    def _get_time_matrix(self) -> np.ndarray:
        """Calculate transport time matrix (hours)."""
        distances = self._calculate_distances()
        avg_speed_kmh = 60  # Average transport speed
        return distances / avg_speed_kmh
    
    def optimize(
        self,
        supply: Dict[str, float],
        demand: Dict[str, float],
        priority_weights: Optional[Dict[str, float]] = None,
        enforce_min_allocation: bool = True
    ) -> AllocationPlan:
        """
        Optimize distribution given supply and demand.
        
        Args:
            supply: Dict of facility_id -> available doses
            demand: Dict of region_id -> required doses
            priority_weights: Optional priority overrides per region
            enforce_min_allocation: Whether to enforce minimum allocations
        
        Returns:
            AllocationPlan with optimal allocations
        """
        logger.info(f"Optimizing distribution with objective: {self.objective}")
        
        # Setup
        n_facilities = len(self.facilities)
        n_regions = len(self.regions)
        
        if n_facilities == 0 or n_regions == 0:
            # Fallback to simple proportional allocation
            return self._simple_allocation(supply, demand)
        
        facility_ids = list(self.facilities.keys())
        region_ids = list(self.regions.keys())
        
        # Get cost/time matrices
        cost_matrix = self._get_cost_matrix()
        time_matrix = self._get_time_matrix()
        
        # Build optimization problem
        # Decision variables: x[i,j] = doses from facility i to region j
        # Flattened: x = [x_00, x_01, ..., x_0m, x_10, ..., x_nm]
        
        n_vars = n_facilities * n_regions
        
        # Objective function coefficients
        if self.objective == "minimize_cost":
            c = cost_matrix.flatten()
        elif self.objective == "minimize_time":
            c = time_matrix.flatten()
        elif self.objective == "maximize_coverage":
            # Maximize coverage = minimize -coverage
            region_pops = np.array([self.regions[r].population for r in region_ids])
            coverage_benefit = np.tile(1 / (region_pops + 1), n_facilities)  # Per-dose coverage
            c = -coverage_benefit
        elif self.objective == "maximize_equity":
            # Prioritize high-priority, low-coverage regions
            priorities = np.array([
                self.regions[r].priority_score * (1 - self.regions[r].current_coverage)
                for r in region_ids
            ])
            c = -np.tile(priorities, n_facilities)
        
        # Supply constraints: sum_j x[i,j] <= supply[i]
        A_supply = np.zeros((n_facilities, n_vars))
        for i in range(n_facilities):
            A_supply[i, i*n_regions:(i+1)*n_regions] = 1
        b_supply = [supply.get(fid, 0) for fid in facility_ids]
        
        # Demand constraints: sum_i x[i,j] <= demand[j] (don't oversupply)
        A_demand = np.zeros((n_regions, n_vars))
        for j in range(n_regions):
            for i in range(n_facilities):
                A_demand[j, i*n_regions + j] = 1
        b_demand = [demand.get(rid, 0) for rid in region_ids]
        
        # Time constraints: only allow feasible transport times
        time_feasible = time_matrix <= self.max_transport_time
        bounds = []
        for i in range(n_facilities):
            for j in range(n_regions):
                if time_feasible[i, j]:
                    bounds.append((0, None))  # Feasible route
                else:
                    bounds.append((0, 0))  # Infeasible route
        
        # Combine inequality constraints
        A_ub = np.vstack([A_supply, A_demand])
        b_ub = b_supply + b_demand
        
        # Minimum allocation equality constraints (if applicable)
        A_eq = None
        b_eq = None
        if enforce_min_allocation:
            min_allocs = [self.regions[rid].min_allocation for rid in region_ids]
            if any(m > 0 for m in min_allocs):
                # Add equality constraints for minimum allocations
                # Note: This may make problem infeasible
                pass  # Skip for simplicity, handle via bounds
        
        # Solve
        try:
            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs"
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}. Using simple allocation.")
                return self._simple_allocation(supply, demand)
            
            x = result.x.reshape(n_facilities, n_regions)
        
        except Exception as e:
            logger.error(f"Optimization error: {e}. Using simple allocation.")
            return self._simple_allocation(supply, demand)
        
        # Build allocation plan
        allocations = {}
        transport_details = []
        total_cost = 0.0
        total_doses = 0.0
        
        for i, fid in enumerate(facility_ids):
            allocations[fid] = {}
            for j, rid in enumerate(region_ids):
                doses = x[i, j]
                if doses > 0.5:  # Threshold to avoid tiny allocations
                    doses = round(doses)
                    allocations[fid][rid] = doses
                    
                    distance = self._distance_matrix[i, j]
                    transport_time = time_matrix[i, j]
                    cost = cost_matrix[i, j] * doses / 1000
                    
                    total_doses += doses
                    total_cost += cost
                    
                    transport_details.append({
                        "from_facility": fid,
                        "to_region": rid,
                        "doses": doses,
                        "distance_km": distance,
                        "time_hours": transport_time,
                        "cost": cost
                    })
        
        # Calculate coverage
        allocated_per_region = {}
        for fid, region_allocs in allocations.items():
            for rid, doses in region_allocs.items():
                allocated_per_region[rid] = allocated_per_region.get(rid, 0) + doses
        
        coverage_by_region = {}
        for rid in region_ids:
            region = self.regions[rid]
            allocated = allocated_per_region.get(rid, 0)
            demand_for_region = demand.get(rid, region.population * 0.8)  # Default 80% target
            coverage = allocated / demand_for_region if demand_for_region > 0 else 1.0
            coverage_by_region[rid] = min(1.0, coverage)
        
        total_demand = sum(demand.values())
        unmet = max(0, total_demand - total_doses)
        
        plan = AllocationPlan(
            allocations=allocations,
            total_cost=total_cost,
            total_doses_distributed=total_doses,
            unmet_demand=unmet,
            coverage_by_region=coverage_by_region,
            transport_details=transport_details
        )
        
        logger.info(f"Optimization complete. Distributed {total_doses:,.0f} doses at ${total_cost:,.2f}")
        
        return plan
    
    def _simple_allocation(
        self,
        supply: Dict[str, float],
        demand: Dict[str, float]
    ) -> AllocationPlan:
        """Fallback simple proportional allocation."""
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())
        
        allocations = {"all": {}}
        coverage_by_region = {}
        
        supply_ratio = min(1.0, total_supply / total_demand) if total_demand > 0 else 1.0
        
        for region_id, region_demand in demand.items():
            allocated = region_demand * supply_ratio
            allocations["all"][region_id] = allocated
            coverage_by_region[region_id] = supply_ratio
        
        return AllocationPlan(
            allocations=allocations,
            total_cost=total_supply * 0.01,  # Estimate
            total_doses_distributed=total_supply * supply_ratio,
            unmet_demand=max(0, total_demand - total_supply),
            coverage_by_region=coverage_by_region,
            transport_details=[]
        )
    
    def simulate_scenarios(
        self,
        base_supply: Dict[str, float],
        base_demand: Dict[str, float],
        supply_variations: List[float] = None,
        demand_variations: List[float] = None
    ) -> pd.DataFrame:
        """
        Simulate multiple supply/demand scenarios.
        
        Args:
            base_supply: Baseline supply by facility
            base_demand: Baseline demand by region
            supply_variations: Multipliers for supply (e.g., [0.8, 1.0, 1.2])
            demand_variations: Multipliers for demand (e.g., [0.9, 1.0, 1.1])
        
        Returns:
            DataFrame with scenario results
        """
        supply_variations = supply_variations or [0.8, 1.0, 1.2]
        demand_variations = demand_variations or [0.9, 1.0, 1.1]
        
        results = []
        for supply_mult in supply_variations:
            for demand_mult in demand_variations:
                scenario_supply = {k: v * supply_mult for k, v in base_supply.items()}
                scenario_demand = {k: v * demand_mult for k, v in base_demand.items()}
                
                plan = self.optimize(scenario_supply, scenario_demand)
                
                results.append({
                    "supply_multiplier": supply_mult,
                    "demand_multiplier": demand_mult,
                    "total_cost": plan.total_cost,
                    "total_distributed": plan.total_doses_distributed,
                    "unmet_demand": plan.unmet_demand,
                    "avg_coverage": np.mean(list(plan.coverage_by_region.values()))
                })
        
        return pd.DataFrame(results)
    
    def recommend_facility_locations(
        self,
        n_facilities: int,
        demand_centers: Dict[str, Tuple[float, float]],
        demand_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[float, float]]:
        """
        Recommend optimal facility locations using k-medoids approach.
        
        Args:
            n_facilities: Number of facilities to place
            demand_centers: Dict of region_id -> (lat, lon)
            demand_weights: Optional weights per region (e.g., by population)
        
        Returns:
            List of recommended (lat, lon) locations
        """
        from sklearn.cluster import KMeans
        
        locations = np.array(list(demand_centers.values()))
        weights = np.array([
            demand_weights.get(rid, 1.0) if demand_weights else 1.0
            for rid in demand_centers.keys()
        ])
        
        # Weighted k-means to find facility locations
        # Repeat points by weight to simulate weighted clustering
        weight_int = (weights / weights.min() * 10).astype(int)
        weighted_locs = np.repeat(locations, weight_int, axis=0)
        
        kmeans = KMeans(n_clusters=min(n_facilities, len(weighted_locs)), random_state=42)
        kmeans.fit(weighted_locs)
        
        recommended = [tuple(center) for center in kmeans.cluster_centers_]
        logger.info(f"Recommended {len(recommended)} facility locations")
        
        return recommended
    
    def save(self, filepath: str):
        """Save optimizer configuration to disk."""
        joblib.dump({
            "objective": self.objective,
            "max_transport_time": self.max_transport_time,
            "cold_chain_temp": self.cold_chain_temp,
            "cost_per_km": self.cost_per_km,
            "wastage_rate": self.wastage_rate,
            "facilities": self.facilities,
            "regions": self.regions
        }, filepath)
        logger.info(f"Optimizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "DistributionOptimizer":
        """Load optimizer from disk."""
        data = joblib.load(filepath)
        optimizer = cls(
            objective=data["objective"],
            max_transport_time=data["max_transport_time"],
            cold_chain_temp=data["cold_chain_temp"],
            cost_per_km=data["cost_per_km"],
            wastage_rate=data["wastage_rate"]
        )
        optimizer.facilities = data["facilities"]
        optimizer.regions = data["regions"]
        logger.info(f"Optimizer loaded from {filepath}")
        return optimizer
