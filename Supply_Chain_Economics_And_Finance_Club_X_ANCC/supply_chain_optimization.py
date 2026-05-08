"""
Supply Chain Demand Forecasting & Optimization
================================================
Comprehensive supply chain optimization system covering:
1. Computational Geometry (zone overlap, convex hull, point-in-polygon)
2. Demand Forecasting (ARIMA, Prophet, Exponential Smoothing)
3. Inventory Optimization (EOQ, Safety Stock, Linear Programming)
4. Facility Location Problems (P-median, Capacitated)
5. Route Planning (Dijkstra, TSP, MST)

Interview Topics Covered:
- Graph algorithms and complexity analysis
- Linear programming and optimization
- Time-series forecasting
- Computational geometry algorithms
- Operations research fundamentals
"""

import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
from datetime import datetime, timedelta  # Date handling
from shapely.geometry import Point, Polygon, MultiPolygon  # Geometric operations
from shapely.ops import unary_union  # Polygon operations
import warnings
warnings.filterwarnings('ignore')  # Clean output

# Try importing optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not installed. Graph algorithms will not be available.")

try:
    from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("Warning: PuLP not installed. Linear programming will not be available.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. ARIMA models will not be available.")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Warning: Prophet not installed. Prophet forecasting will not be available.")


class ComputationalGeometry:
    """
    Computational Geometry Algorithms for Logistics
    ================================================
    Solves spatial problems in supply chain management using geometric algorithms.
    
    Applications:
    - Delivery zone overlap detection
    - Service area analysis
    - Warehouse location optimization
    - Closest facility finding
    
    Interview Concepts:
    - Time/Space complexity analysis
    - Divide and conquer algorithms
    - Convex hull properties
    - Ray casting for point location
    """
    
    @staticmethod
    def convex_hull_graham_scan(points):
        """
        Graham Scan Algorithm for Convex Hull
        --------------------------------------
        Finds the smallest convex polygon containing all points.
        
        Algorithm Steps:
        1. Find lowest point (or leftmost if tie) as anchor
        2. Sort points by polar angle relative to anchor
        3. Process points, maintaining only left turns
        
        Time Complexity: O(n log n) - dominated by sorting
        Space Complexity: O(n) - storing hull points
        
        Parameters:
        -----------
        points : list - List of (x, y) coordinate tuples
        
        Returns:
        --------
        list - Points forming convex hull in counter-clockwise order
        
        Interview Questions:
        --------------------
        Q: Why is convex hull useful in logistics?
        A: Defines minimum bounding region for delivery zones,
           helps identify outlier locations, used in facility placement.
        
        Q: What's the cross product doing here?
        A: Tests if three points make a left turn (counter-clockwise).
           Positive = left turn, Negative = right turn, Zero = collinear
        
        Q: Can you improve this?
        A: For very large datasets, use parallel algorithms or
           Chan's algorithm (O(n log h) where h = hull size)
        """
        def cross_product(o, a, b):
            """
            Calculate cross product to determine turn direction
            
            Returns:
            - Positive: Counter-clockwise turn (left)
            - Negative: Clockwise turn (right)  
            - Zero: Points are collinear
            """
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Sort points lexicographically (first by x, then by y)
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        # Build lower hull
        lower = []
        for p in points:
            # Remove points that make clockwise turn
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # Concatenate L and U (remove duplicate endpoints)
        return lower[:-1] + upper[:-1]
    
    @staticmethod
    def point_in_polygon(point, polygon):
        """
        Check if point is inside polygon using ray casting
        Args:
            point: (x, y) tuple
            polygon: List of (x, y) tuples forming polygon
        Returns:
            Boolean indicating if point is inside
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def polygon_intersection(poly1, poly2):
        """
        Calculate intersection area of two polygons using Shapely
        Args:
            poly1, poly2: Lists of (x, y) tuples
        Returns:
            Intersection area
        """
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        
        intersection = p1.intersection(p2)
        return intersection.area
    
    @staticmethod
    def zone_overlap_analysis(zones):
        """
        Analyze overlapping delivery zones
        Args:
            zones: Dictionary of zone_name -> list of (x, y) coordinates
        Returns:
            DataFrame with overlap analysis
        """
        zone_names = list(zones.keys())
        n = len(zone_names)
        
        results = []
        for i in range(n):
            for j in range(i + 1, n):
                poly1 = Polygon(zones[zone_names[i]])
                poly2 = Polygon(zones[zone_names[j]])
                
                if poly1.intersects(poly2):
                    overlap_area = ComputationalGeometry.polygon_intersection(
                        zones[zone_names[i]], zones[zone_names[j]]
                    )
                    results.append({
                        'Zone_1': zone_names[i],
                        'Zone_2': zone_names[j],
                        'Overlap_Area': overlap_area,
                        'Zone_1_Area': poly1.area,
                        'Zone_2_Area': poly2.area,
                        'Overlap_Pct_1': (overlap_area / poly1.area) * 100,
                        'Overlap_Pct_2': (overlap_area / poly2.area) * 100
                    })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def closest_pair_of_points(points):
        """
        Find closest pair of points using divide and conquer
        Args:
            points: List of (x, y) tuples
        Returns:
            Tuple of (point1, point2, distance)
        """
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def brute_force(points):
            min_dist = float('inf')
            pair = None
            n = len(points)
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(points[i], points[j])
                    if d < min_dist:
                        min_dist = d
                        pair = (points[i], points[j])
            return pair, min_dist
        
        if len(points) <= 3:
            return brute_force(points)
        
        # Sort by x-coordinate
        points_sorted = sorted(points, key=lambda p: p[0])
        
        # Divide
        mid = len(points_sorted) // 2
        left_pair, left_dist = ComputationalGeometry.closest_pair_of_points(points_sorted[:mid])
        right_pair, right_dist = ComputationalGeometry.closest_pair_of_points(points_sorted[mid:])
        
        # Find minimum
        if left_dist < right_dist:
            min_dist = left_dist
            min_pair = left_pair
        else:
            min_dist = right_dist
            min_pair = right_pair
        
        return min_pair, min_dist


class RoutePlanning:
    """Route planning and optimization using graph algorithms"""
    
    @staticmethod
    def create_distance_matrix(locations):
        """
        Create distance matrix from location coordinates
        Args:
            locations: Dictionary of location_name -> (x, y)
        Returns:
            Distance matrix as numpy array
        """
        names = list(locations.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    loc1 = locations[names[i]]
                    loc2 = locations[names[j]]
                    matrix[i][j] = np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
        
        return matrix, names
    
    @staticmethod
    def dijkstra_shortest_path(graph, start, end):
        """
        Find shortest path using Dijkstra's algorithm
        Args:
            graph: Dictionary of node -> {neighbor: distance}
            start: Start node
            end: End node
        Returns:
            Tuple of (path, total_distance)
        """
        if not HAS_NETWORKX:
            print("NetworkX not available. Install: pip install networkx")
            return None, None
        
        G = nx.DiGraph()
        for node, neighbors in graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)
        
        try:
            path = nx.dijkstra_path(G, start, end, weight='weight')
            distance = nx.dijkstra_path_length(G, start, end, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    @staticmethod
    def traveling_salesman_nearest_neighbor(distance_matrix, start_index=0):
        """
        Solve TSP using nearest neighbor heuristic
        Args:
            distance_matrix: 2D numpy array of distances
            start_index: Starting city index
        Returns:
            Tour and total distance
        """
        n = len(distance_matrix)
        unvisited = set(range(n))
        current = start_index
        tour = [current]
        unvisited.remove(current)
        total_distance = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            total_distance += distance_matrix[current][nearest]
            current = nearest
            tour.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_distance += distance_matrix[current][start_index]
        tour.append(start_index)
        
        return tour, total_distance
    
    @staticmethod
    def minimum_spanning_tree(graph):
        """
        Find MST using Kruskal's algorithm
        Args:
            graph: List of (node1, node2, weight) tuples
        Returns:
            List of edges in MST
        """
        if not HAS_NETWORKX:
            print("NetworkX not available")
            return None
        
        G = nx.Graph()
        for node1, node2, weight in graph:
            G.add_edge(node1, node2, weight=weight)
        
        mst = nx.minimum_spanning_tree(G, weight='weight')
        return list(mst.edges(data=True))


class DemandForecasting:
    """Demand forecasting models"""
    
    @staticmethod
    def generate_sample_demand_data(periods=365, trend=True, seasonality=True, noise_level=0.1):
        """Generate sample demand data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # Base demand
        demand = 100 * np.ones(periods)
        
        # Add trend
        if trend:
            demand += np.linspace(0, 50, periods)
        
        # Add seasonality
        if seasonality:
            seasonal = 20 * np.sin(2 * np.pi * np.arange(periods) / 30)
            demand += seasonal
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.mean(demand), periods)
        demand += noise
        
        df = pd.DataFrame({'date': dates, 'demand': demand})
        return df
    
    @staticmethod
    def moving_average_forecast(data, window=7, forecast_periods=30):
        """Simple moving average forecast"""
        ma = data['demand'].rolling(window=window).mean()
        
        # Forecast using last MA value
        last_ma = ma.iloc[-1]
        forecast = [last_ma] * forecast_periods
        
        return forecast
    
    @staticmethod
    def exponential_smoothing_forecast(data, forecast_periods=30):
        """Exponential smoothing forecast"""
        if not HAS_STATSMODELS:
            print("Exponential smoothing not available. Install statsmodels")
            return None
        
        model = ExponentialSmoothing(data['demand'], seasonal_periods=7, 
                                     trend='add', seasonal='add')
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_periods)
        
        return forecast.values
    
    @staticmethod
    def arima_forecast(data, order=(2, 1, 2), forecast_periods=30):
        """ARIMA forecast"""
        if not HAS_STATSMODELS:
            print("ARIMA not available. Install statsmodels")
            return None
        
        model = ARIMA(data['demand'], order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_periods)
        
        return forecast.values
    
    @staticmethod
    def prophet_forecast(data, forecast_periods=30):
        """Facebook Prophet forecast"""
        if not HAS_PROPHET:
            print("Prophet not available. Install: pip install prophet")
            return None
        
        df = data.copy()
        df.columns = ['ds', 'y']
        
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        return forecast['yhat'].tail(forecast_periods).values
    
    @staticmethod
    def evaluate_forecast(actual, predicted):
        """Calculate forecast accuracy metrics"""
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }


class InventoryOptimization:
    """Inventory management and optimization"""
    
    @staticmethod
    def economic_order_quantity(annual_demand, ordering_cost, holding_cost):
        """
        Calculate EOQ
        Args:
            annual_demand: Annual demand quantity
            ordering_cost: Cost per order
            holding_cost: Annual holding cost per unit
        Returns:
            Optimal order quantity
        """
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return eoq
    
    @staticmethod
    def reorder_point(daily_demand, lead_time, safety_stock=0):
        """
        Calculate reorder point
        Args:
            daily_demand: Average daily demand
            lead_time: Lead time in days
            safety_stock: Safety stock quantity
        Returns:
            Reorder point
        """
        rop = (daily_demand * lead_time) + safety_stock
        return rop
    
    @staticmethod
    def safety_stock(daily_demand, demand_std, lead_time, service_level=0.95):
        """
        Calculate safety stock
        Args:
            daily_demand: Average daily demand
            demand_std: Standard deviation of demand
            lead_time: Lead time in days
            service_level: Desired service level (0-1)
        Returns:
            Safety stock quantity
        """
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        return safety_stock
    
    @staticmethod
    def optimize_inventory_pulp(products, budget_constraint=None):
        """
        Optimize inventory levels using linear programming
        Args:
            products: List of dicts with 'demand', 'holding_cost', 'stockout_cost'
            budget_constraint: Maximum budget for inventory
        Returns:
            Optimal inventory levels
        """
        if not HAS_PULP:
            print("PuLP not available. Install: pip install pulp")
            return None
        
        n = len(products)
        
        # Create problem
        prob = LpProblem("Inventory_Optimization", LpMinimize)
        
        # Decision variables
        inventory = [LpVariable(f"inv_{i}", lowBound=0) for i in range(n)]
        stockouts = [LpVariable(f"stockout_{i}", lowBound=0) for i in range(n)]
        
        # Objective: Minimize total cost
        prob += lpSum([
            products[i]['holding_cost'] * inventory[i] + 
            products[i]['stockout_cost'] * stockouts[i]
            for i in range(n)
        ])
        
        # Constraints
        for i in range(n):
            prob += inventory[i] + stockouts[i] == products[i]['demand']
        
        if budget_constraint:
            prob += lpSum([products[i]['holding_cost'] * inventory[i] for i in range(n)]) <= budget_constraint
        
        # Solve
        prob.solve()
        
        results = {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'inventory_levels': [value(inventory[i]) for i in range(n)],
            'stockouts': [value(stockouts[i]) for i in range(n)]
        }
        
        return results


class FacilityLocation:
    """Facility location and logistics network optimization"""
    
    @staticmethod
    def p_median_problem(customer_locations, potential_facilities, p, demands=None):
        """
        Solve p-median facility location problem
        Args:
            customer_locations: List of (x, y) tuples
            potential_facilities: List of (x, y) tuples
            p: Number of facilities to open
            demands: List of demand values (optional)
        Returns:
            Selected facility indices and assignments
        """
        if not HAS_PULP:
            print("PuLP not available")
            return None
        
        n_customers = len(customer_locations)
        n_facilities = len(potential_facilities)
        
        if demands is None:
            demands = [1] * n_customers
        
        # Calculate distances
        distances = np.zeros((n_customers, n_facilities))
        for i in range(n_customers):
            for j in range(n_facilities):
                distances[i][j] = np.sqrt(
                    (customer_locations[i][0] - potential_facilities[j][0])**2 +
                    (customer_locations[i][1] - potential_facilities[j][1])**2
                )
        
        # Create problem
        prob = LpProblem("P_Median", LpMinimize)
        
        # Decision variables
        y = [LpVariable(f"facility_{j}", cat='Binary') for j in range(n_facilities)]
        x = [[LpVariable(f"assign_{i}_{j}", cat='Binary') 
              for j in range(n_facilities)] for i in range(n_customers)]
        
        # Objective: Minimize total weighted distance
        prob += lpSum([
            demands[i] * distances[i][j] * x[i][j]
            for i in range(n_customers) for j in range(n_facilities)
        ])
        
        # Constraints
        # Open exactly p facilities
        prob += lpSum(y) == p
        
        # Each customer assigned to exactly one facility
        for i in range(n_customers):
            prob += lpSum([x[i][j] for j in range(n_facilities)]) == 1
        
        # Can only assign to open facilities
        for i in range(n_customers):
            for j in range(n_facilities):
                prob += x[i][j] <= y[j]
        
        # Solve
        prob.solve()
        
        selected_facilities = [j for j in range(n_facilities) if value(y[j]) == 1]
        assignments = [
            [j for j in range(n_facilities) if value(x[i][j]) == 1][0]
            for i in range(n_customers)
        ]
        
        return selected_facilities, assignments, value(prob.objective)
    
    @staticmethod
    def capacitated_facility_location(customer_demands, facility_costs, facility_capacities, 
                                     transportation_costs):
        """
        Solve capacitated facility location problem
        Args:
            customer_demands: List of demand values
            facility_costs: List of fixed costs for opening facilities
            facility_capacities: List of facility capacities
            transportation_costs: 2D array of transportation costs
        Returns:
            Solution with opened facilities and allocations
        """
        if not HAS_PULP:
            print("PuLP not available")
            return None
        
        n_customers = len(customer_demands)
        n_facilities = len(facility_costs)
        
        # Create problem
        prob = LpProblem("Capacitated_Facility_Location", LpMinimize)
        
        # Decision variables
        y = [LpVariable(f"open_{j}", cat='Binary') for j in range(n_facilities)]
        x = [[LpVariable(f"flow_{i}_{j}", lowBound=0) 
              for j in range(n_facilities)] for i in range(n_customers)]
        
        # Objective
        prob += lpSum([facility_costs[j] * y[j] for j in range(n_facilities)]) + \
                lpSum([transportation_costs[i][j] * x[i][j] 
                      for i in range(n_customers) for j in range(n_facilities)])
        
        # Constraints
        # Satisfy all demand
        for i in range(n_customers):
            prob += lpSum([x[i][j] for j in range(n_facilities)]) == customer_demands[i]
        
        # Capacity constraints
        for j in range(n_facilities):
            prob += lpSum([x[i][j] for i in range(n_customers)]) <= facility_capacities[j] * y[j]
        
        # Solve
        prob.solve()
        
        results = {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'opened_facilities': [j for j in range(n_facilities) if value(y[j]) == 1],
            'allocations': [[value(x[i][j]) for j in range(n_facilities)] for i in range(n_customers)]
        }
        
        return results


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 80)
    print("SUPPLY CHAIN DEMAND FORECASTING & OPTIMIZATION")
    print("=" * 80)
    
    # 1. Computational Geometry - Zone Overlap Analysis
    print("\n" + "=" * 80)
    print("PART 1: COMPUTATIONAL GEOMETRY - ZONE OVERLAP ANALYSIS")
    print("=" * 80)
    
    zones = {
        'Zone_A': [(0, 0), (10, 0), (10, 10), (0, 10)],
        'Zone_B': [(5, 5), (15, 5), (15, 15), (5, 15)],
        'Zone_C': [(8, 8), (18, 8), (18, 18), (8, 18)]
    }
    
    overlap_df = ComputationalGeometry.zone_overlap_analysis(zones)
    print("\nZone Overlap Analysis:")
    print(overlap_df)
    
    # Convex Hull
    points = [(0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3)]
    hull = ComputationalGeometry.convex_hull_graham_scan(points)
    print(f"\nConvex Hull of {len(points)} points: {hull}")
    
    # 2. Demand Forecasting
    print("\n" + "=" * 80)
    print("PART 2: DEMAND FORECASTING")
    print("=" * 80)
    
    # Generate sample data
    demand_data = DemandForecasting.generate_sample_demand_data(periods=180)
    print(f"\nGenerated {len(demand_data)} days of demand data")
    print(demand_data.head())
    
    # Moving Average
    ma_forecast = DemandForecasting.moving_average_forecast(demand_data, window=7, forecast_periods=30)
    print(f"\nMoving Average Forecast (7-day): {ma_forecast[:5]}...")
    
    # ARIMA
    if HAS_STATSMODELS:
        arima_forecast = DemandForecasting.arima_forecast(demand_data, forecast_periods=30)
        print(f"ARIMA Forecast: {arima_forecast[:5]}...")
    
    # Prophet
    if HAS_PROPHET:
        prophet_forecast = DemandForecasting.prophet_forecast(demand_data, forecast_periods=30)
        print(f"Prophet Forecast: {prophet_forecast[:5]}...")
    
    # 3. Inventory Optimization
    print("\n" + "=" * 80)
    print("PART 3: INVENTORY OPTIMIZATION")
    print("=" * 80)
    
    # EOQ Calculation
    annual_demand = 10000
    ordering_cost = 50
    holding_cost = 2
    
    eoq = InventoryOptimization.economic_order_quantity(annual_demand, ordering_cost, holding_cost)
    print(f"\nEconomic Order Quantity (EOQ):")
    print(f"Annual Demand: {annual_demand} units")
    print(f"Ordering Cost: ${ordering_cost}")
    print(f"Holding Cost: ${holding_cost}/unit/year")
    print(f"Optimal Order Quantity: {eoq:.0f} units")
    
    # Reorder Point
    daily_demand = annual_demand / 365
    lead_time = 7  # days
    rop = InventoryOptimization.reorder_point(daily_demand, lead_time, safety_stock=50)
    print(f"\nReorder Point: {rop:.0f} units")
    
    # Linear Programming Optimization
    if HAS_PULP:
        products = [
            {'demand': 100, 'holding_cost': 2, 'stockout_cost': 10},
            {'demand': 150, 'holding_cost': 3, 'stockout_cost': 15},
            {'demand': 200, 'holding_cost': 1.5, 'stockout_cost': 8}
        ]
        
        result = InventoryOptimization.optimize_inventory_pulp(products, budget_constraint=1000)
        print(f"\nInventory Optimization Results:")
        print(f"Status: {result['status']}")
        print(f"Total Cost: ${result['total_cost']:.2f}")
        print(f"Inventory Levels: {[f'{x:.1f}' for x in result['inventory_levels']]}")
    
    # 4. Route Planning
    print("\n" + "=" * 80)
    print("PART 4: ROUTE PLANNING & OPTIMIZATION")
    print("=" * 80)
    
    # TSP using Nearest Neighbor
    locations = {
        'Warehouse': (0, 0),
        'Customer_1': (2, 3),
        'Customer_2': (5, 1),
        'Customer_3': (6, 6),
        'Customer_4': (8, 3)
    }
    
    distance_matrix, names = RoutePlanning.create_distance_matrix(locations)
    tour, total_distance = RoutePlanning.traveling_salesman_nearest_neighbor(distance_matrix, 0)
    
    print(f"\nTraveling Salesman Problem (Nearest Neighbor):")
    print(f"Tour: {' -> '.join([names[i] for i in tour])}")
    print(f"Total Distance: {total_distance:.2f}")
    
    # 5. Facility Location
    print("\n" + "=" * 80)
    print("PART 5: FACILITY LOCATION OPTIMIZATION")
    print("=" * 80)
    
    if HAS_PULP:
        customer_locations = [(1, 2), (3, 4), (5, 1), (7, 6), (2, 8)]
        potential_facilities = [(2, 3), (5, 5), (8, 2), (4, 7)]
        demands = [10, 15, 20, 25, 12]
        
        selected, assignments, obj_value = FacilityLocation.p_median_problem(
            customer_locations, potential_facilities, p=2, demands=demands
        )
        
        print(f"\nP-Median Problem (p=2):")
        print(f"Selected Facilities: {selected}")
        print(f"Customer Assignments: {assignments}")
        print(f"Total Weighted Distance: {obj_value:.2f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
