"""
Low-Emission Equitable Urban Mobility Plan
Analysis and optimization of urban transportation systems for sustainability and equity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("Warning: PuLP not installed. Optimization models will not be available.")

try:
    from scipy.optimize import minimize, linprog
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not installed. Some optimization features will not be available.")


class TransportDataAnalysis:
    """Analyze urban transport data to identify patterns and issues"""
    
    @staticmethod
    def generate_sample_transport_data(n_corridors=50, n_zones=20):
        """Generate sample urban transport data for analysis"""
        np.random.seed(42)
        
        # Transport corridors data
        corridors = pd.DataFrame({
            'corridor_id': range(1, n_corridors + 1),
            'length_km': np.random.uniform(2, 15, n_corridors),
            'daily_vehicles': np.random.randint(5000, 50000, n_corridors),
            'avg_speed_kmh': np.random.uniform(15, 60, n_corridors),
            'public_transit_coverage': np.random.uniform(0, 1, n_corridors),
            'bike_lane_present': np.random.choice([0, 1], n_corridors, p=[0.6, 0.4]),
            'pedestrian_infrastructure': np.random.uniform(0, 1, n_corridors),
            'avg_income_level': np.random.choice(['Low', 'Medium', 'High'], n_corridors),
            'population_density': np.random.uniform(1000, 15000, n_corridors)
        })
        
        # Calculate emissions (simplified model)
        # Emission factor depends on vehicle type, speed, etc.
        corridors['co2_tons_per_day'] = (
            corridors['daily_vehicles'] * corridors['length_km'] * 
            (0.12 + (60 - corridors['avg_speed_kmh']) * 0.002)  # Higher emissions at lower speeds
        ) / 1000
        
        # Accessibility score (0-100)
        corridors['accessibility_score'] = (
            corridors['public_transit_coverage'] * 40 +
            corridors['bike_lane_present'] * 20 +
            corridors['pedestrian_infrastructure'] * 20 +
            (corridors['avg_speed_kmh'] / 60) * 20
        )
        
        # Zone-level data
        zones = pd.DataFrame({
            'zone_id': range(1, n_zones + 1),
            'population': np.random.randint(10000, 100000, n_zones),
            'jobs': np.random.randint(5000, 80000, n_zones),
            'avg_income': np.random.uniform(30000, 120000, n_zones),
            'transit_stops': np.random.randint(2, 20, n_zones),
            'transit_frequency_per_hour': np.random.randint(2, 15, n_zones),
            'car_ownership_rate': np.random.uniform(0.3, 0.9, n_zones),
            'center_x': np.random.uniform(0, 50, n_zones),
            'center_y': np.random.uniform(0, 50, n_zones)
        })
        
        zones['jobs_housing_balance'] = zones['jobs'] / zones['population']
        zones['transit_accessibility'] = (
            zones['transit_stops'] * zones['transit_frequency_per_hour']
        )
        
        return corridors, zones
    
    @staticmethod
    def identify_high_emission_corridors(corridors_df, threshold_percentile=75):
        """Identify high-emission corridors"""
        emission_threshold = corridors_df['co2_tons_per_day'].quantile(threshold_percentile / 100)
        
        high_emission = corridors_df[corridors_df['co2_tons_per_day'] >= emission_threshold].copy()
        high_emission['emission_category'] = 'High'
        
        print(f"\nHigh Emission Corridors (>{threshold_percentile}th percentile):")
        print(f"Threshold: {emission_threshold:.2f} tons CO2/day")
        print(f"Number of corridors: {len(high_emission)}")
        print(f"Total daily emissions: {high_emission['co2_tons_per_day'].sum():.2f} tons CO2")
        print(f"Average emissions: {high_emission['co2_tons_per_day'].mean():.2f} tons CO2/day")
        
        return high_emission
    
    @staticmethod
    def identify_low_accessibility_corridors(corridors_df, threshold_percentile=25):
        """Identify corridors with low accessibility"""
        access_threshold = corridors_df['accessibility_score'].quantile(threshold_percentile / 100)
        
        low_access = corridors_df[corridors_df['accessibility_score'] <= access_threshold].copy()
        low_access['accessibility_category'] = 'Low'
        
        print(f"\nLow Accessibility Corridors (<{threshold_percentile}th percentile):")
        print(f"Threshold: {access_threshold:.2f}")
        print(f"Number of corridors: {len(low_access)}")
        print(f"Average accessibility score: {low_access['accessibility_score'].mean():.2f}")
        
        return low_access
    
    @staticmethod
    def equity_analysis(corridors_df):
        """Analyze transport equity by income level"""
        equity_stats = corridors_df.groupby('avg_income_level').agg({
            'accessibility_score': 'mean',
            'public_transit_coverage': 'mean',
            'bike_lane_present': 'mean',
            'co2_tons_per_day': 'mean',
            'corridor_id': 'count'
        }).round(2)
        
        equity_stats.columns = ['Avg_Accessibility', 'Transit_Coverage', 
                               'Bike_Lane_Rate', 'Avg_Emissions', 'Corridor_Count']
        
        print("\nEquity Analysis by Income Level:")
        print(equity_stats)
        
        return equity_stats
    
    @staticmethod
    def calculate_modal_share(zones_df):
        """Calculate current modal share based on zone characteristics"""
        # Simplified modal share model
        zones = zones_df.copy()
        
        # Private vehicle share (based on car ownership and transit access)
        zones['private_vehicle_share'] = (
            zones['car_ownership_rate'] * 0.7 + 
            (1 - zones['transit_accessibility'] / zones['transit_accessibility'].max()) * 0.3
        )
        
        # Public transit share
        zones['public_transit_share'] = (
            (zones['transit_accessibility'] / zones['transit_accessibility'].max()) * 0.6 +
            (1 - zones['car_ownership_rate']) * 0.4
        ) * 0.5
        
        # Active transport (walking/cycling)
        zones['active_transport_share'] = 1 - zones['private_vehicle_share'] - zones['public_transit_share']
        zones['active_transport_share'] = zones['active_transport_share'].clip(lower=0)
        
        # Normalize
        total = zones[['private_vehicle_share', 'public_transit_share', 'active_transport_share']].sum(axis=1)
        zones['private_vehicle_share'] /= total
        zones['public_transit_share'] /= total
        zones['active_transport_share'] /= total
        
        return zones
    
    @staticmethod
    def visualize_corridor_analysis(corridors_df, save_path=None):
        """Visualize corridor emissions vs accessibility"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Emissions vs Accessibility
        axes[0, 0].scatter(corridors_df['accessibility_score'], 
                          corridors_df['co2_tons_per_day'],
                          c=corridors_df['daily_vehicles'], 
                          cmap='YlOrRd', alpha=0.6, s=100)
        axes[0, 0].set_xlabel('Accessibility Score', fontsize=11)
        axes[0, 0].set_ylabel('CO2 Emissions (tons/day)', fontsize=11)
        axes[0, 0].set_title('Emissions vs Accessibility', fontsize=13)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Emissions by income level
        income_order = ['Low', 'Medium', 'High']
        corridors_df['avg_income_level'] = pd.Categorical(
            corridors_df['avg_income_level'], categories=income_order, ordered=True
        )
        corridors_df.boxplot(column='co2_tons_per_day', by='avg_income_level', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Income Level', fontsize=11)
        axes[0, 1].set_ylabel('CO2 Emissions (tons/day)', fontsize=11)
        axes[0, 1].set_title('Emissions by Income Level', fontsize=13)
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)
        
        # Accessibility by income level
        corridors_df.boxplot(column='accessibility_score', by='avg_income_level', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Income Level', fontsize=11)
        axes[1, 0].set_ylabel('Accessibility Score', fontsize=11)
        axes[1, 0].set_title('Accessibility by Income Level', fontsize=13)
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)
        
        # Transit coverage
        transit_data = corridors_df.groupby('avg_income_level')['public_transit_coverage'].mean()
        axes[1, 1].bar(range(len(transit_data)), transit_data.values, 
                      color=['#e74c3c', '#f39c12', '#27ae60'])
        axes[1, 1].set_xticks(range(len(transit_data)))
        axes[1, 1].set_xticklabels(transit_data.index)
        axes[1, 1].set_xlabel('Income Level', fontsize=11)
        axes[1, 1].set_ylabel('Avg Transit Coverage', fontsize=11)
        axes[1, 1].set_title('Public Transit Coverage by Income', fontsize=13)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TransitOptimization:
    """Optimize public transit routes and schedules"""
    
    @staticmethod
    def optimize_bus_routes(zones_df, num_routes=5, budget=1000000):
        """
        Optimize bus route network using linear programming
        Args:
            zones_df: DataFrame with zone information
            num_routes: Number of bus routes to design
            budget: Budget constraint
        Returns:
            Optimized route assignments
        """
        if not HAS_PULP:
            print("PuLP not available. Install: pip install pulp")
            return None
        
        n_zones = len(zones_df)
        
        # Create problem
        prob = LpProblem("Bus_Route_Optimization", LpMaximize)
        
        # Decision variables: whether zone i is served by route r
        x = [[LpVariable(f"zone_{i}_route_{r}", cat='Binary') 
              for r in range(num_routes)] for i in range(n_zones)]
        
        # Route operational variables
        route_active = [LpVariable(f"route_{r}_active", cat='Binary') 
                       for r in range(num_routes)]
        
        # Objective: Maximize population served weighted by need
        # Need is higher for low-income, low-transit-access areas
        zones_df['transit_need'] = (
            zones_df['population'] / 
            (zones_df['transit_accessibility'] + 1)  # +1 to avoid division by zero
        )
        
        prob += lpSum([
            zones_df.iloc[i]['transit_need'] * x[i][r]
            for i in range(n_zones) for r in range(num_routes)
        ])
        
        # Constraints
        # Each zone served by at most one route
        for i in range(n_zones):
            prob += lpSum([x[i][r] for r in range(num_routes)]) <= 1
        
        # Route connectivity (simplified: each route serves at least 3 zones if active)
        for r in range(num_routes):
            prob += lpSum([x[i][r] for i in range(n_zones)]) >= 3 * route_active[r]
            prob += lpSum([x[i][r] for i in range(n_zones)]) <= 15 * route_active[r]
        
        # Budget constraint (cost per route and per zone served)
        route_fixed_cost = 50000
        zone_service_cost = 5000
        
        prob += lpSum([
            route_fixed_cost * route_active[r] + 
            zone_service_cost * lpSum([x[i][r] for i in range(n_zones)])
            for r in range(num_routes)
        ]) <= budget
        
        # Solve
        prob.solve()
        
        # Extract solution
        active_routes = [r for r in range(num_routes) if value(route_active[r]) == 1]
        route_assignments = {}
        
        for r in active_routes:
            zones_in_route = [i for i in range(n_zones) if value(x[i][r]) == 1]
            route_assignments[f'Route_{r+1}'] = zones_in_route
        
        results = {
            'status': LpStatus[prob.status],
            'objective_value': value(prob.objective),
            'active_routes': len(active_routes),
            'route_assignments': route_assignments,
            'zones_served': sum(len(zones) for zones in route_assignments.values())
        }
        
        return results
    
    @staticmethod
    def optimize_bus_frequency(route_demand, budget_hours, min_frequency=2, max_frequency=20):
        """
        Optimize bus frequency for routes based on demand
        Args:
            route_demand: Dictionary of route -> demand
            budget_hours: Total bus-hours available per day
            min_frequency: Minimum buses per hour
            max_frequency: Maximum buses per hour
        Returns:
            Optimal frequency allocation
        """
        if not HAS_PULP:
            print("PuLP not available")
            return None
        
        routes = list(route_demand.keys())
        n_routes = len(routes)
        
        # Create problem
        prob = LpProblem("Bus_Frequency_Optimization", LpMaximize)
        
        # Decision variables: frequency for each route
        freq = [LpVariable(f"freq_{r}", lowBound=min_frequency, upBound=max_frequency, cat='Integer') 
                for r in range(n_routes)]
        
        # Objective: Maximize demand served (with diminishing returns)
        # Service quality = demand * sqrt(frequency)
        prob += lpSum([
            route_demand[routes[r]] * freq[r]
            for r in range(n_routes)
        ])
        
        # Constraint: Total bus-hours
        prob += lpSum([freq[r] * 18 for r in range(n_routes)]) <= budget_hours  # 18 hours operation
        
        # Solve
        prob.solve()
        
        results = {
            'status': LpStatus[prob.status],
            routes[r]: int(value(freq[r]))
            for r in range(n_routes)
        }
        
        return results


class EVInfrastructure:
    """Electric Vehicle infrastructure planning and optimization"""
    
    @staticmethod
    def optimize_ev_charging_stations(zones_df, num_stations, budget=500000):
        """
        Optimize placement of EV charging stations
        Args:
            zones_df: DataFrame with zone information
            num_stations: Number of charging stations to place
            budget: Budget constraint
        Returns:
            Optimal station locations
        """
        if not HAS_PULP:
            print("PuLP not available")
            return None
        
        n_zones = len(zones_df)
        
        # Calculate station priority (based on population, income, current infrastructure)
        zones_df['ev_priority'] = (
            zones_df['population'] * 0.4 +
            (zones_df['avg_income'] / zones_df['avg_income'].max()) * 1000 * 0.3 +
            zones_df['car_ownership_rate'] * 1000 * 0.3
        )
        
        # Create problem
        prob = LpProblem("EV_Charging_Station_Placement", LpMaximize)
        
        # Decision variables
        station = [LpVariable(f"station_{i}", cat='Binary') for i in range(n_zones)]
        
        # Coverage variables (zone j is covered by station in zone i if within distance threshold)
        # Calculate distance matrix
        coords = zones_df[['center_x', 'center_y']].values
        distances = cdist(coords, coords, metric='euclidean') if HAS_SCIPY else np.zeros((n_zones, n_zones))
        
        coverage_threshold = 5  # km
        coverage = [[LpVariable(f"cover_{i}_{j}", cat='Binary') 
                    for j in range(n_zones)] for i in range(n_zones)]
        
        # Objective: Maximize weighted coverage
        prob += lpSum([
            zones_df.iloc[j]['ev_priority'] * coverage[i][j]
            for i in range(n_zones) for j in range(n_zones)
        ])
        
        # Constraints
        # Place exactly num_stations
        prob += lpSum(station) == num_stations
        
        # Coverage only if station exists and within distance
        for i in range(n_zones):
            for j in range(n_zones):
                prob += coverage[i][j] <= station[i]
                if HAS_SCIPY and distances[i][j] > coverage_threshold:
                    prob += coverage[i][j] == 0
        
        # Each zone covered by at most one station (simplification)
        for j in range(n_zones):
            prob += lpSum([coverage[i][j] for i in range(n_zones)]) <= 1
        
        # Budget constraint (cost per station varies by zone)
        station_cost = 50000
        prob += lpSum([station_cost * station[i] for i in range(n_zones)]) <= budget
        
        # Solve
        prob.solve()
        
        selected_zones = [i for i in range(n_zones) if value(station[i]) == 1]
        
        results = {
            'status': LpStatus[prob.status],
            'selected_zones': selected_zones,
            'zone_ids': zones_df.iloc[selected_zones]['zone_id'].tolist(),
            'total_coverage': value(prob.objective)
        }
        
        return results
    
    @staticmethod
    def ev_adoption_projection(current_vehicles, years=10, growth_rate=0.25):
        """
        Project EV adoption over time
        Args:
            current_vehicles: Current number of vehicles
            years: Projection period
            growth_rate: Annual EV adoption growth rate
        Returns:
            DataFrame with projections
        """
        projections = []
        
        for year in range(years + 1):
            # Logistic growth model (S-curve)
            max_adoption = 0.7  # 70% maximum EV penetration
            ev_share = max_adoption / (1 + np.exp(-growth_rate * (year - 5)))
            
            ev_vehicles = int(current_vehicles * ev_share)
            ice_vehicles = current_vehicles - ev_vehicles
            
            # Emission reduction (EVs have ~70% lower emissions considering electricity mix)
            emission_reduction = ev_vehicles * 0.7
            
            projections.append({
                'year': year,
                'ev_share': ev_share,
                'ev_vehicles': ev_vehicles,
                'ice_vehicles': ice_vehicles,
                'emission_reduction_index': emission_reduction / current_vehicles
            })
        
        return pd.DataFrame(projections)


class MultiObjectiveOptimization:
    """Multi-objective optimization for mobility planning"""
    
    @staticmethod
    def optimize_mobility_plan(corridors_df, weights={'emissions': 0.4, 'equity': 0.3, 'travel_time': 0.3}):
        """
        Multi-objective optimization balancing emissions, equity, and travel time
        Args:
            corridors_df: Corridor data
            weights: Weights for different objectives
        Returns:
            Optimized intervention plan
        """
        if not HAS_PULP:
            print("PuLP not available")
            return None
        
        n_corridors = len(corridors_df)
        
        # Create problem
        prob = LpProblem("Multi_Objective_Mobility_Plan", LpMaximize)
        
        # Decision variables: intervention types for each corridor
        # 0: No intervention, 1: Bus lane, 2: Bike lane, 3: Both
        transit_upgrade = [LpVariable(f"transit_{i}", cat='Binary') for i in range(n_corridors)]
        bike_lane = [LpVariable(f"bike_{i}", cat='Binary') for i in range(n_corridors)]
        
        # Normalize objectives (0-1 scale)
        max_emissions = corridors_df['co2_tons_per_day'].max()
        max_access_need = (100 - corridors_df['accessibility_score']).max()
        
        # Objective components
        # 1. Emission reduction (priority to high-emission corridors)
        emission_reduction = lpSum([
            (corridors_df.iloc[i]['co2_tons_per_day'] / max_emissions) * 
            (transit_upgrade[i] * 0.3 + bike_lane[i] * 0.15)  # Reduction factors
            for i in range(n_corridors)
        ])
        
        # 2. Equity improvement (priority to low-access, low-income areas)
        equity_score = lpSum([
            ((100 - corridors_df.iloc[i]['accessibility_score']) / max_access_need) *
            (1.5 if corridors_df.iloc[i]['avg_income_level'] == 'Low' else 1.0) *
            (transit_upgrade[i] * 0.4 + bike_lane[i] * 0.2)
            for i in range(n_corridors)
        ])
        
        # 3. Travel time improvement (transit upgrades reduce congestion)
        time_improvement = lpSum([
            (corridors_df.iloc[i]['daily_vehicles'] / corridors_df['daily_vehicles'].max()) *
            transit_upgrade[i] * 0.25
            for i in range(n_corridors)
        ])
        
        # Combined weighted objective
        prob += (
            weights['emissions'] * emission_reduction +
            weights['equity'] * equity_score +
            weights['travel_time'] * time_improvement
        )
        
        # Constraints
        budget = 50  # Million dollars
        transit_cost = 2  # Million per corridor
        bike_cost = 0.5  # Million per corridor
        
        prob += lpSum([
            transit_upgrade[i] * transit_cost + bike_lane[i] * bike_cost
            for i in range(n_corridors)
        ]) <= budget
        
        # Cannot have bike lane without some space (simplified)
        for i in range(n_corridors):
            if corridors_df.iloc[i]['avg_speed_kmh'] > 50:  # High-speed corridors
                prob += bike_lane[i] == 0
        
        # Solve
        prob.solve()
        
        # Extract results
        interventions = []
        for i in range(n_corridors):
            transit = value(transit_upgrade[i])
            bike = value(bike_lane[i])
            
            if transit or bike:
                intervention_type = []
                if transit:
                    intervention_type.append('Transit Upgrade')
                if bike:
                    intervention_type.append('Bike Lane')
                
                interventions.append({
                    'corridor_id': corridors_df.iloc[i]['corridor_id'],
                    'interventions': ', '.join(intervention_type),
                    'current_emissions': corridors_df.iloc[i]['co2_tons_per_day'],
                    'current_accessibility': corridors_df.iloc[i]['accessibility_score'],
                    'income_level': corridors_df.iloc[i]['avg_income_level']
                })
        
        results = {
            'status': LpStatus[prob.status],
            'objective_value': value(prob.objective),
            'interventions': pd.DataFrame(interventions),
            'total_corridors_upgraded': len(interventions)
        }
        
        return results


class ModalShiftSimulation:
    """Simulate modal shift scenarios"""
    
    @staticmethod
    def simulate_scenario(zones_df, interventions, years=10):
        """
        Simulate modal shift based on interventions
        Args:
            zones_df: Zone data with current modal shares
            interventions: Dictionary of intervention parameters
            years: Simulation period
        Returns:
            DataFrame with modal share projections
        """
        # Calculate current modal shares
        zones = TransportDataAnalysis.calculate_modal_share(zones_df)
        
        current_shares = {
            'Private Vehicle': zones['private_vehicle_share'].mean(),
            'Public Transit': zones['public_transit_share'].mean(),
            'Active Transport': zones['active_transport_share'].mean()
        }
        
        # Intervention effects (annual shift rates)
        transit_improvement = interventions.get('transit_upgrade', 0)  # 0-1
        bike_infrastructure = interventions.get('bike_lanes', 0)  # 0-1
        ev_adoption = interventions.get('ev_rate', 0.2)  # Annual growth
        
        # Simulation
        projections = []
        
        for year in range(years + 1):
            # Modal shift (simplified model)
            # Transit improvements attract car users
            transit_shift = transit_improvement * 0.02 * year
            # Bike lanes attract short car trips
            bike_shift = bike_infrastructure * 0.015 * year
            
            # Calculate new shares
            private_share = max(0.2, current_shares['Private Vehicle'] - transit_shift - bike_shift)
            transit_share = min(0.5, current_shares['Public Transit'] + transit_shift + bike_shift * 0.3)
            active_share = 1 - private_share - transit_share
            
            # EV penetration in private vehicles
            ev_penetration = min(0.7, ev_adoption * year)
            
            # Emissions calculation (baseline: 100 index)
            # Private vehicles: 100, Transit: 30, Active: 0, EVs: 30
            emission_index = (
                private_share * (1 - ev_penetration) * 100 +
                private_share * ev_penetration * 30 +
                transit_share * 30 +
                active_share * 0
            )
            
            projections.append({
                'year': year,
                'private_vehicle_share': private_share,
                'public_transit_share': transit_share,
                'active_transport_share': active_share,
                'ev_penetration': ev_penetration,
                'emission_index': emission_index
            })
        
        return pd.DataFrame(projections)
    
    @staticmethod
    def visualize_modal_shift(baseline, scenario1, scenario2=None, save_path=None):
        """Visualize modal shift scenarios"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Modal share over time
        years = baseline['year'].values
        
        axes[0].plot(years, baseline['private_vehicle_share'] * 100, 
                    label='Baseline - Private Vehicle', linestyle='--', marker='o')
        axes[0].plot(years, scenario1['private_vehicle_share'] * 100, 
                    label='Scenario 1 - Private Vehicle', marker='s')
        if scenario2 is not None:
            axes[0].plot(years, scenario2['private_vehicle_share'] * 100, 
                        label='Scenario 2 - Private Vehicle', marker='^')
        
        axes[0].plot(years, baseline['public_transit_share'] * 100, 
                    label='Baseline - Public Transit', linestyle='--', marker='o', alpha=0.6)
        axes[0].plot(years, scenario1['public_transit_share'] * 100, 
                    label='Scenario 1 - Public Transit', marker='s', alpha=0.6)
        
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Modal Share (%)', fontsize=12)
        axes[0].set_title('Modal Share Projections', fontsize=14)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Emissions over time
        axes[1].plot(years, baseline['emission_index'], 
                    label='Baseline', linestyle='--', marker='o', linewidth=2)
        axes[1].plot(years, scenario1['emission_index'], 
                    label='Scenario 1', marker='s', linewidth=2)
        if scenario2 is not None:
            axes[1].plot(years, scenario2['emission_index'], 
                        label='Scenario 2', marker='^', linewidth=2)
        
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Emission Index (Baseline=100)', fontsize=12)
        axes[1].set_title('Emission Reduction Projections', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("LOW-EMISSION EQUITABLE URBAN MOBILITY PLAN")
    print("=" * 80)
    
    # Generate sample data
    print("\nGenerating sample urban transport data...")
    corridors_df, zones_df = TransportDataAnalysis.generate_sample_transport_data(
        n_corridors=50, n_zones=20
    )
    
    # 1. Data Analysis
    print("\n" + "=" * 80)
    print("PART 1: TRANSPORT DATA ANALYSIS")
    print("=" * 80)
    
    # Identify problem corridors
    high_emission = TransportDataAnalysis.identify_high_emission_corridors(corridors_df)
    low_access = TransportDataAnalysis.identify_low_accessibility_corridors(corridors_df)
    
    # Equity analysis
    equity_stats = TransportDataAnalysis.equity_analysis(corridors_df)
    
    # Visualize
    print("\nGenerating corridor analysis visualizations...")
    TransportDataAnalysis.visualize_corridor_analysis(corridors_df)
    
    # 2. Transit Optimization
    print("\n" + "=" * 80)
    print("PART 2: PUBLIC TRANSIT OPTIMIZATION")
    print("=" * 80)
    
    if HAS_PULP:
        bus_routes = TransitOptimization.optimize_bus_routes(zones_df, num_routes=5, budget=1000000)
        print(f"\nBus Route Optimization:")
        print(f"Status: {bus_routes['status']}")
        print(f"Active Routes: {bus_routes['active_routes']}")
        print(f"Zones Served: {bus_routes['zones_served']} out of {len(zones_df)}")
        
        # Frequency optimization
        route_demand = {f'Route_{i+1}': np.random.randint(1000, 5000) 
                       for i in range(bus_routes['active_routes'])}
        frequencies = TransitOptimization.optimize_bus_frequency(route_demand, budget_hours=500)
        print(f"\nOptimal Bus Frequencies:")
        for route, freq in frequencies.items():
            if route != 'status':
                print(f"{route}: {freq} buses/hour")
    
    # 3. EV Infrastructure
    print("\n" + "=" * 80)
    print("PART 3: EV INFRASTRUCTURE PLANNING")
    print("=" * 80)
    
    if HAS_PULP and HAS_SCIPY:
        ev_stations = EVInfrastructure.optimize_ev_charging_stations(
            zones_df, num_stations=8, budget=500000
        )
        print(f"\nEV Charging Station Placement:")
        print(f"Status: {ev_stations['status']}")
        print(f"Selected Zones: {ev_stations['zone_ids']}")
    
    # EV adoption projection
    ev_projection = EVInfrastructure.ev_adoption_projection(
        current_vehicles=100000, years=10, growth_rate=0.3
    )
    print(f"\nEV Adoption Projection (10 years):")
    print(ev_projection[['year', 'ev_share', 'ev_vehicles']].head())
    
    # 4. Multi-Objective Optimization
    print("\n" + "=" * 80)
    print("PART 4: MULTI-OBJECTIVE MOBILITY PLAN")
    print("=" * 80)
    
    if HAS_PULP:
        mobility_plan = MultiObjectiveOptimization.optimize_mobility_plan(
            corridors_df,
            weights={'emissions': 0.4, 'equity': 0.3, 'travel_time': 0.3}
        )
        
        print(f"\nMobility Plan Optimization:")
        print(f"Status: {mobility_plan['status']}")
        print(f"Objective Value: {mobility_plan['objective_value']:.3f}")
        print(f"Corridors Upgraded: {mobility_plan['total_corridors_upgraded']}")
        print(f"\nTop Interventions:")
        print(mobility_plan['interventions'].head(10))
    
    # 5. Modal Shift Simulation
    print("\n" + "=" * 80)
    print("PART 5: MODAL SHIFT SIMULATION")
    print("=" * 80)
    
    # Baseline scenario (no interventions)
    baseline = ModalShiftSimulation.simulate_scenario(
        zones_df, 
        {'transit_upgrade': 0, 'bike_lanes': 0, 'ev_rate': 0.1},
        years=10
    )
    
    # Scenario 1: Moderate interventions
    scenario1 = ModalShiftSimulation.simulate_scenario(
        zones_df,
        {'transit_upgrade': 0.6, 'bike_lanes': 0.5, 'ev_rate': 0.25},
        years=10
    )
    
    # Scenario 2: Aggressive interventions
    scenario2 = ModalShiftSimulation.simulate_scenario(
        zones_df,
        {'transit_upgrade': 0.9, 'bike_lanes': 0.8, 'ev_rate': 0.35},
        years=10
    )
    
    print("\nModal Shift Projections:")
    print("\nBaseline (Year 10):")
    print(f"Private Vehicle: {baseline.iloc[-1]['private_vehicle_share']*100:.1f}%")
    print(f"Public Transit: {baseline.iloc[-1]['public_transit_share']*100:.1f}%")
    print(f"Emission Index: {baseline.iloc[-1]['emission_index']:.1f}")
    
    print("\nScenario 1 - Moderate (Year 10):")
    print(f"Private Vehicle: {scenario1.iloc[-1]['private_vehicle_share']*100:.1f}%")
    print(f"Public Transit: {scenario1.iloc[-1]['public_transit_share']*100:.1f}%")
    print(f"Emission Index: {scenario1.iloc[-1]['emission_index']:.1f}")
    print(f"Emission Reduction: {(1 - scenario1.iloc[-1]['emission_index']/baseline.iloc[-1]['emission_index'])*100:.1f}%")
    
    # Visualize
    print("\nGenerating modal shift visualizations...")
    ModalShiftSimulation.visualize_modal_shift(baseline, scenario1, scenario2)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Recommendations:")
    print("1. Prioritize transit improvements in low-income, high-emission corridors")
    print("2. Implement bike lane network in medium-density areas")
    print("3. Deploy EV charging stations in high-car-ownership zones")
    print("4. Target 30-40% emission reduction through integrated interventions")
    print("5. Monitor equity metrics to ensure benefits reach underserved communities")
