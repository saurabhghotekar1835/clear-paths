#!/usr/bin/env python3
"""
Integrate AQI data with routing system using spatial interpolation
This approach keeps AQI data independent and interpolates values for road segments
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
import json

def load_aqi_data():
    """Load AQI data from CSV file."""
    try:
        df = pd.read_csv("gurugram_air_quality_with_aqi.csv")
        print(f"✅ Loaded AQI data: {len(df)} readings from {df['station_name'].nunique()} stations")
        return df
    except Exception as e:
        print(f"❌ Error loading AQI data: {e}")
        return None

def find_nearest_aqi_stations(road_lat: float, road_lon: float, aqi_df: pd.DataFrame, 
                             max_distance_km: float = 5.0, max_stations: int = 5):
    """
    Find nearest AQI stations to a road point using inverse distance weighting.
    
    Args:
        road_lat, road_lon: Road point coordinates
        aqi_df: AQI data DataFrame
        max_distance_km: Maximum distance to consider (default 5km)
        max_stations: Maximum number of stations to use (default 5)
    
    Returns:
        List of (station_name, distance_km, aqi_value) tuples
    """
    
    # Calculate distances from road point to all stations
    station_coords = aqi_df[['latitude', 'longitude']].values
    road_coords = np.array([[road_lat, road_lon]])
    
    # Calculate distances in km (approximate)
    distances = cdist(road_coords, station_coords, metric='euclidean') * 111  # Convert to km
    distances = distances[0]  # Flatten
    
    # Filter stations within max distance
    valid_indices = distances <= max_distance_km
    
    if not any(valid_indices):
        print(f"⚠️ No AQI stations within {max_distance_km}km of ({road_lat}, {road_lon})")
        return []
    
    # Get valid stations with distances and AQI values
    valid_stations = []
    for i, is_valid in enumerate(valid_indices):
        if is_valid:
            station_name = aqi_df.iloc[i]['station_name']
            distance_km = distances[i]
            aqi_value = aqi_df.iloc[i]['AQI']
            valid_stations.append((station_name, distance_km, aqi_value))
    
    # Sort by distance and take top stations
    valid_stations.sort(key=lambda x: x[1])
    return valid_stations[:max_stations]

def interpolate_aqi(road_lat: float, road_lon: float, aqi_df: pd.DataFrame, 
                   time_slot: str = "00:00-02:59") -> float:
    """
    Interpolate AQI value for a road point using inverse distance weighting.
    
    Args:
        road_lat, road_lon: Road point coordinates
        aqi_df: AQI data DataFrame
        time_slot: Time slot to use (default: "00:00-02:59")
    
    Returns:
        Interpolated AQI value
    """
    
    # Filter AQI data for the specific time slot
    time_data = aqi_df[aqi_df['time_slot'] == time_slot].copy()
    
    if len(time_data) == 0:
        print(f"⚠️ No AQI data for time slot: {time_slot}")
        return 100.0  # Default moderate AQI
    
    # Find nearest stations
    nearest_stations = find_nearest_aqi_stations(road_lat, road_lon, time_data)
    
    if not nearest_stations:
        return 100.0  # Default moderate AQI
    
    # Calculate inverse distance weighted AQI
    total_weight = 0
    weighted_sum = 0
    
    for station_name, distance_km, aqi_value in nearest_stations:
        if distance_km == 0:
            return aqi_value  # Exact match
        
        # Inverse distance weight (1/distance^2)
        weight = 1 / (distance_km ** 2)
        total_weight += weight
        weighted_sum += weight * aqi_value
    
    if total_weight == 0:
        return 100.0  # Default moderate AQI
    
    interpolated_aqi = weighted_sum / total_weight
    return round(interpolated_aqi)

def update_backend_with_aqi():
    """Update the backend to use AQI data in routing."""
    
    print("🔄 Updating backend with AQI integration...")
    
    # Read the current backend
    try:
        with open("backend.py", "r", encoding="utf-8") as f:
            backend_code = f.read()
    except Exception as e:
        print(f"❌ Error reading backend.py: {e}")
        return False
    
    # Add AQI imports and functions
    aqi_imports = """
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
"""
    
    aqi_functions = """

# Global AQI data
aqi_data = None

def load_aqi_data():
    \"\"\"Load AQI data from CSV file.\"\"\"
    global aqi_data
    try:
        aqi_data = pd.read_csv("gurugram_air_quality_with_aqi.csv")
        print(f"✅ Loaded AQI data: {len(aqi_data)} readings from {aqi_data['station_name'].nunique()} stations")
        return True
    except Exception as e:
        print(f"❌ Error loading AQI data: {e}")
        return False

def interpolate_aqi_for_point(lat: float, lon: float, time_slot: str = "00:00-02:59") -> float:
    \"\"\"Interpolate AQI value for a road point.\"\"\"
    global aqi_data
    
    if aqi_data is None:
        return 100.0  # Default moderate AQI
    
    # Filter for time slot
    time_data = aqi_data[aqi_data['time_slot'] == time_slot].copy()
    
    if len(time_data) == 0:
        return 100.0  # Default moderate AQI
    
    # Calculate distances
    station_coords = time_data[['latitude', 'longitude']].values
    road_coords = np.array([[lat, lon]])
    distances = cdist(road_coords, station_coords, metric='euclidean') * 111  # Convert to km
    distances = distances[0]
    
    # Find stations within 10km
    valid_indices = distances <= 10.0
    
    if not any(valid_indices):
        return 100.0  # Default moderate AQI
    
    # Calculate inverse distance weighted AQI
    total_weight = 0
    weighted_sum = 0
    
    for i, is_valid in enumerate(valid_indices):
        if is_valid:
            distance_km = distances[i]
            aqi_value = time_data.iloc[i]['AQI']
            
            if distance_km == 0:
                return aqi_value  # Exact match
            
            weight = 1 / (distance_km ** 2)
            total_weight += weight
            weighted_sum += weight * aqi_value
    
    if total_weight == 0:
        return 100.0  # Default moderate AQI
    
    return round(weighted_sum / total_weight)

def calculate_road_cost_with_aqi(distance_km: float, road_lat: float, road_lon: float, 
                                time_slot: str = "00:00-02:59") -> float:
    \"\"\"Calculate road cost using distance and interpolated AQI.\"\"\"
    
    # Get interpolated AQI for this road point
    aqi_value = interpolate_aqi_for_point(road_lat, road_lon, time_slot)
    
    # AQI weight factor (higher AQI = higher cost)
    aqi_weight = 1 + (aqi_value / 100.0)  # AQI 100 = 2x cost, AQI 200 = 3x cost
    
    # Calculate total cost
    cost = distance_km * aqi_weight
    
    return cost
"""
    
    # Update the calculate_road_cost function
    updated_calculate_road_cost = """
def calculate_road_cost(distance_km: float, pm25: float) -> float:
    \"\"\"Calculate road cost based on distance and PM2.5 (legacy function).\"\"\"
    # Legacy function - kept for compatibility
    return distance_km * (1 + pm25 / 50.0)
"""
    
    # Update the build_graph function to use AQI
    updated_build_graph = """
def build_graph():
    \"\"\"Build a NetworkX graph from the road data using ALL coordinate points with AQI.\"\"\"
    global road_graph, road_data
    
    print("Building road network graph with ALL coordinate points and AQI integration...")
    
    road_graph = nx.Graph()
    
    for road in road_data:
        coordinates = road['coordinates']
        
        # Create nodes for ALL coordinate points in this road
        road_nodes = []
        for i, coord in enumerate(coordinates):
            lon, lat = coord
            node_id = f"{lat:.6f}_{lon:.6f}"
            road_nodes.append(node_id)
            
            # Store node coordinates for later use
            node_coordinates[node_id] = (lat, lon)
        
        # Connect consecutive nodes in the road
        for i in range(len(road_nodes) - 1):
            current_node = road_nodes[i]
            next_node = road_nodes[i + 1]
            
            # Calculate distance between these two points
            lat1, lon1 = coordinates[i][1], coordinates[i][0]
            lat2, lon2 = coordinates[i + 1][1], coordinates[i + 1][0]
            segment_distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111  # Convert to km
            
            # Calculate midpoint for AQI interpolation
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            
            # Calculate custom cost based on distance and interpolated AQI
            cost = calculate_road_cost_with_aqi(segment_distance, mid_lat, mid_lon)
            
            # Add edge with attributes
            road_graph.add_edge(
                current_node, 
                next_node,
                distance=segment_distance,
                cost=cost,
                road_id=road['road_id'],
                road_name=road['road_name'],
                coordinates=[coordinates[i], coordinates[i + 1]]
            )
    
    print(f"Graph built with {road_graph.number_of_nodes()} nodes and {road_graph.number_of_edges()} edges")
    print(f"Using ALL coordinate points with AQI integration!")
"""
    
    # Update the main section to load AQI data
    updated_main = """
if __name__ == "__main__":
    print("Starting Clear Paths Backend with AQI Integration...")
    
    # Load road data
    load_road_data()
    
    # Load AQI data
    if not load_aqi_data():
        print("⚠️ AQI data not available, using default values")
    
    # Build graph
    build_graph()
    
    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    
    # Apply updates to backend code
    updated_backend = backend_code.replace(
        "import json\nimport math\nimport heapq\nimport os",
        "import json\nimport math\nimport heapq\nimport os" + aqi_imports
    )
    
    updated_backend = updated_backend.replace(
        "def calculate_road_cost(distance_km: float, pm25: float) -> float:",
        aqi_functions + "\n" + updated_calculate_road_cost
    )
    
    updated_backend = updated_backend.replace(
        "def build_graph():",
        updated_build_graph
    )
    
    updated_backend = updated_backend.replace(
        "if __name__ == \"__main__\":",
        updated_main
    )
    
    # Write updated backend
    try:
        with open("backend.py", "w", encoding="utf-8") as f:
            f.write(updated_backend)
        print("✅ Backend updated with AQI integration!")
        return True
    except Exception as e:
        print(f"❌ Error updating backend: {e}")
        return False

def test_aqi_integration():
    """Test AQI interpolation with sample road points."""
    
    print("🧪 Testing AQI integration...")
    
    # Load AQI data
    aqi_df = load_aqi_data()
    if aqi_df is None:
        return
    
    # Test points in Gurugram
    test_points = [
        (28.4595, 77.0266, "Cyber City area"),
        (28.4089, 77.0418, "Sohna Road area"),
        (28.5022, 77.4055, "Manesar area")
    ]
    
    print("\n📍 AQI Interpolation Test:")
    print("=" * 50)
    
    for lat, lon, description in test_points:
        aqi_value = interpolate_aqi(lat, lon, aqi_df)
        print(f"{description}: ({lat}, {lon}) → AQI: {aqi_value}")

if __name__ == "__main__":
    print("🌬️ AQI-Routing Integration")
    print("=" * 40)
    
    # Test AQI integration
    test_aqi_integration()
    print()
    
    # Update backend
    if update_backend_with_aqi():
        print("\n✅ Integration completed! Backend now uses AQI data for routing.")
        print("\n🚀 Next steps:")
        print("   1. Install scipy: pip install scipy")
        print("   2. Restart backend: python backend.py")
        print("   3. Test routing with AQI-aware paths")
    else:
        print("\n❌ Integration failed!") 