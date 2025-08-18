import json
import math
import heapq
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import pickle
from datetime import datetime

from typing import List, Dict, Tuple, Optional
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from config import settings

# Initialize FastAPI app
app = FastAPI(title="Air Quality Routing API", description="API for finding routes with minimal air pollution exposure")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the graph and data
road_graph = None
road_data = None
node_coordinates = {}
aqi_data = None
aqi_cache = {}  # Cache for AQI interpolation results

# Graph persistence files
GRAPH_CACHE_FILE = settings.GRAPH_CACHE_FILE
GRAPH_METADATA_FILE = settings.GRAPH_METADATA_FILE

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    time_slot: Optional[str] = "00:00-02:59"  # Default time slot

class RouteResponse(BaseModel):
    route: List[List[float]]  # List of [lon, lat] coordinates
    total_distance: float
    total_pollution: float
    message: str

def get_file_modification_time(filepath: str) -> float:
    """Get the modification time of a file."""
    try:
        return os.path.getmtime(filepath)
    except OSError:
        return 0

def should_rebuild_graph() -> bool:
    """Check if the graph needs to be rebuilt based on file modifications."""
    if not os.path.exists(GRAPH_CACHE_FILE) or not os.path.exists(GRAPH_METADATA_FILE):
        print("📁 Graph cache files not found, will rebuild graph")
        return True
    
    try:
        with open(GRAPH_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Check if source files have been modified since last build
        geojson_file = "gurugram_roads_real.geojson"
        csv_file = "gurugram_air_quality_with_aqi.csv"
        
        geojson_time = get_file_modification_time(geojson_file)
        csv_time = get_file_modification_time(csv_file)
        last_build_time = metadata.get('last_build_time', 0)
        
        if geojson_time > last_build_time or csv_time > last_build_time:
            print("📝 Source files modified since last build, will rebuild graph")
            return True
        
        print("✅ Graph cache is up to date, loading from disk")
        return False
        
    except Exception as e:
        print(f"⚠️ Error checking graph metadata: {e}, will rebuild graph")
        return True

def save_graph_to_disk():
    """Save the built graph and metadata to disk."""
    global road_graph, node_coordinates, aqi_cache
    
    try:
        # Save the graph
        with open(GRAPH_CACHE_FILE, 'wb') as f:
            pickle.dump({
                'graph': road_graph,
                'node_coordinates': node_coordinates,
                'aqi_cache': aqi_cache
            }, f)
        
        # Save metadata
        metadata = {
            'last_build_time': datetime.now().timestamp(),
            'nodes_count': road_graph.number_of_nodes() if road_graph else 0,
            'edges_count': road_graph.number_of_edges() if road_graph else 0,
            'build_date': datetime.now().isoformat()
        }
        
        with open(GRAPH_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Graph saved to disk: {road_graph.number_of_nodes():,} nodes, {road_graph.number_of_edges():,} edges")
        
    except Exception as e:
        print(f"❌ Error saving graph to disk: {e}")

def load_graph_from_disk() -> bool:
    """Load the graph from disk if available."""
    global road_graph, node_coordinates, aqi_cache
    
    try:
        with open(GRAPH_CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        
        road_graph = data['graph']
        node_coordinates = data['node_coordinates']
        aqi_cache = data.get('aqi_cache', {})
        
        print(f"📂 Graph loaded from disk: {road_graph.number_of_nodes():,} nodes, {road_graph.number_of_edges():,} edges")
        print(f"🔄 AQI cache loaded: {len(aqi_cache)} cached values")
        return True
        
    except Exception as e:
        print(f"❌ Error loading graph from disk: {e}")
        return False

def load_road_data():
    """Load and parse the GeoJSON file containing road segments (without fake PM2.5 data)."""
    global road_data, road_graph, node_coordinates
    
    print("Loading road data from GeoJSON file...")
    
    try:
        # Try to load the new comprehensive road data first
        geojson_file = "gurugram_roads_real.geojson"
        if not os.path.exists(geojson_file):
            # Fall back to original file
            geojson_file = "gurugram_road_segments_with_air_quality.geojson"
            print(f"⚠️ New road data not found, using original: {geojson_file}")
        else:
            print(f"✅ Using new comprehensive road data: {geojson_file}")
        
        # Read the GeoJSON file
        gdf = gpd.read_file(geojson_file)
        
        # Convert to dictionary for easier processing
        road_data = []
        node_coordinates = {}
        
        for idx, row in gdf.iterrows():
            # Handle different property names between old and new data
            if 'road_name' in row:
                road_name = row['road_name']
            elif 'name' in row:
                road_name = row['name']
            else:
                road_name = 'Unnamed Road'
            
            if 'distance_km' in row:
                distance_km = row['distance_km']
            elif 'length' in row:
                distance_km = row['length']
            else:
                distance_km = 1.0  # Default value
            
            # Get coordinates
            coords = list(row.geometry.coords)
            if len(coords) >= 2:
                start_lat, start_lng = coords[0][1], coords[0][0]
                end_lat, end_lng = coords[-1][1], coords[-1][0]
                
                feature = {
                    'road_id': f"road_{idx}",
                    'road_name': road_name,
                    'road_type': row.get('highway', 'unknown'),
                    'distance_km': distance_km,
                    'start_lat': start_lat,
                    'start_lng': start_lng,
                    'end_lat': end_lat,
                    'end_lng': end_lng,
                    'coordinates': coords
                }
                
                road_data.append(feature)
                
                # Store node coordinates for later use
                start_node = f"{start_lat:.6f}_{start_lng:.6f}"
                end_node = f"{end_lat:.6f}_{end_lng:.6f}"
                
                node_coordinates[start_node] = (start_lat, start_lng)
                node_coordinates[end_node] = (end_lat, end_lng)
        
        print(f"✅ Loaded {len(road_data)} road segments from {geojson_file}")
        print(f"🌬️ Road data loaded without fake PM2.5 - will use AQI interpolation for routing")
        
    except Exception as e:
        print(f"❌ Error loading road data: {e}")
        raise


def build_graph():
    """Build a NetworkX graph from the road data using ALL coordinate points with AQI."""
    global road_graph, road_data, aqi_cache
    
    print("Building road network graph with ALL coordinate points and AQI integration...")
    print(f"📊 Processing {len(road_data)} road segments...")
    
    # Clear AQI cache for fresh start
    aqi_cache.clear()
    print("🧹 Cleared AQI cache for fresh interpolation...")
    
    road_graph = nx.DiGraph()  # Use directed graph for one-way road support
    total_nodes = 0
    total_edges = 0
    total_coordinates = 0
    oneway_roads = 0
    bidirectional_roads = 0
    
    # Progress tracking
    progress_interval = max(1, len(road_data) // 20)  # Show progress every 5%
    
    for road_idx, road in enumerate(road_data):
        # Show progress
        if road_idx % progress_interval == 0:
            progress = (road_idx / len(road_data)) * 100
            cache_size = len(aqi_cache)
            print(f"🔄 Progress: {progress:.1f}% - Road {road_idx}/{len(road_data)} - Nodes: {total_nodes}, Edges: {total_edges}, AQI Cache: {cache_size}")
        
        coordinates = road['coordinates']
        total_coordinates += len(coordinates)
        
        # Determine road direction based on road type and name patterns (once per road)
        is_oneway = determine_road_direction(road['road_name'], road['road_type'])
        
        # Track road types
        if is_oneway:
            oneway_roads += 1
        else:
            bidirectional_roads += 1
        
        # Create nodes for ALL coordinate points in this road
        road_nodes = []
        for i, coord in enumerate(coordinates):
            lon, lat = coord
            node_id = f"{lat:.6f}_{lon:.6f}"
            road_nodes.append(node_id)
            
            # Store node coordinates for later use
            node_coordinates[node_id] = (lat, lon)
        
        # Connect consecutive nodes in the road with directional support
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
            
            # Add forward edge (always present)
            road_graph.add_edge(
                current_node, 
                next_node,
                distance=segment_distance,
                road_id=road['road_id'],
                road_name=road['road_name'],
                coordinates=[coordinates[i], coordinates[i + 1]],
                direction='forward'
            )
            total_edges += 1
            
            # Add reverse edge only if road is bidirectional
            if not is_oneway:
                road_graph.add_edge(
                    next_node,
                    current_node, 
                    distance=segment_distance,
                    road_id=road['road_id'],
                    road_name=road['road_name'],
                    coordinates=[coordinates[i + 1], coordinates[i]],  # Reversed coordinates
                    direction='reverse'
                )
                total_edges += 1
        
        total_nodes = len(road_graph.nodes)
    
    print(f"✅ Graph built successfully!")
    print(f"📊 Final stats: {road_graph.number_of_nodes():,} nodes and {road_graph.number_of_edges():,} edges")
    print(f"📈 Total coordinates processed: {total_coordinates:,}")
    print(f"🔄 Node deduplication: {total_coordinates:,} coordinates → {road_graph.number_of_nodes():,} unique nodes")
    print(f"🛣️ Road direction analysis: {oneway_roads:,} one-way roads, {bidirectional_roads:,} bidirectional roads")
    print(f"🚀 Graph built with directional support and real-time AQI integration - ready for realistic routing!")
    print(f"💾 AQI cache size: {len(aqi_cache)} interpolated values")



def determine_road_direction(road_name: str, road_type: str) -> bool:
    """Determine if a road is one-way based on name patterns and road type."""
    
    # Convert to lowercase for pattern matching
    name_lower = road_name.lower()
    type_lower = road_type.lower()
    
    # One-way indicators in road names
    oneway_patterns = [
        'underpass', 'overpass', 'flyover', 'ramp', 'slip road',
        'service road', 'link road', 'connector', 'approach',
        'exit', 'entry', 'loop', 'roundabout', 'circle'
    ]
    
    # Check for one-way patterns in name
    for pattern in oneway_patterns:
        if pattern in name_lower:
            return True
    
    # Road types that are typically one-way
    oneway_types = [
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
        'ramp', 'slip_road', 'service', 'track'
    ]
    
    if type_lower in oneway_types:
        return True
    
    # Major highways and expressways often have divided lanes (treat as one-way per direction)
    if type_lower in ['motorway', 'trunk'] and any(word in name_lower for word in ['expressway', 'highway', 'freeway']):
        return True
    
    # Default to bidirectional
    return False


def load_aqi_data():
    """Load AQI data from CSV file."""
    global aqi_data
    try:
        aqi_data = pd.read_csv("gurugram_air_quality_with_aqi.csv")
        print(f"✅ Loaded AQI data: {len(aqi_data)} readings from {aqi_data['station_name'].nunique()} stations")
        return True
    except Exception as e:
        print(f"❌ Error loading AQI data: {e}")
        return False

def interpolate_aqi_for_point(lat: float, lon: float, time_slot: str = "00:00-02:59") -> float:
    """Interpolate AQI value for a road point with aggressive caching and simplified calculation."""
    global aqi_data, aqi_cache
    
    if aqi_data is None:
        return 100.0  # Default moderate AQI
    
    # Create cache key with reduced precision for better cache hits
    cache_key = f"{lat:.3f}_{lon:.3f}_{time_slot}"
    
    # Check cache first
    if cache_key in aqi_cache:
        return aqi_cache[cache_key]
    
    # Filter for time slot
    time_data = aqi_data[aqi_data['time_slot'] == time_slot].copy()
    
    if len(time_data) == 0:
        result = 100.0  # Default moderate AQI
    else:
        # Simplified approach: find closest station within reasonable distance
        station_coords = time_data[['latitude', 'longitude']].values
        road_coords = np.array([[lat, lon]])
        distances = cdist(road_coords, station_coords, metric='euclidean') * 111  # Convert to km
        distances = distances[0]
        
        # Find closest station within 10km
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance <= 10.0:
            # Use closest station's AQI with distance-based interpolation
            closest_aqi = time_data.iloc[min_distance_idx]['AQI']
            
            # Find second closest for simple interpolation
            second_closest_idx = np.argsort(distances)[1] if len(distances) > 1 else min_distance_idx
            second_distance = distances[second_closest_idx]
            
            if second_distance <= 10.0 and len(distances) > 1:
                second_aqi = time_data.iloc[second_closest_idx]['AQI']
                # Simple weighted average of two closest stations
                total_weight = (1/min_distance) + (1/second_distance)
                result = ((closest_aqi/min_distance) + (second_aqi/second_distance)) / total_weight
            else:
                result = closest_aqi
        else:
            result = 100.0  # Default moderate AQI
    
    # Cache the result
    aqi_cache[cache_key] = round(result)
    return aqi_cache[cache_key]

def calculate_road_cost_with_aqi(distance_km: float, road_lat: float, road_lon: float, 
                                time_slot: str = "00:00-02:59") -> float:
    """Calculate road cost using distance and interpolated AQI with balanced penalties for effective routing."""
    
    # Get interpolated AQI for this road point
    aqi_value = interpolate_aqi_for_point(road_lat, road_lon, time_slot)
    
    # Balanced AQI weight factor - stronger penalties to encourage cleaner routes
    if aqi_value <= 50:  # Good - No penalty
        aqi_weight = 1.0
    elif aqi_value <= 100:  # Satisfactory - Small penalty
        aqi_weight = 1.5
    elif aqi_value <= 150:  # Moderate - Moderate penalty
        aqi_weight = 3.0
    elif aqi_value <= 200:  # Unhealthy - Higher penalty
        aqi_weight = 6.0
    elif aqi_value <= 300:  # Very Unhealthy - High penalty
        aqi_weight = 12.0
    else:  # Hazardous - Maximum penalty
        aqi_weight = 20.0
    
    # Calculate total cost
    cost = distance_km * aqi_weight
    
    return cost

def calculate_road_cost_distance_only(distance_km: float, road_lat: float, road_lon: float, 
                                     time_slot: str = "00:00-02:59") -> float:
    """Calculate road cost using only distance (ignoring AQI)."""
    # For distance-only routing, cost = distance (no AQI penalty)
    return distance_km

def get_current_time_slot() -> str:
    """Get the current time slot based on the current time."""
    current_hour = datetime.now().hour
    
    if 0 <= current_hour < 3:
        return "00:00-02:59"
    elif 3 <= current_hour < 6:
        return "03:00-05:59"
    elif 6 <= current_hour < 9:
        return "06:00-08:59"
    elif 9 <= current_hour < 12:
        return "09:00-11:59"
    elif 12 <= current_hour < 15:
        return "12:00-14:59"
    elif 15 <= current_hour < 18:
        return "15:00-17:59"
    elif 18 <= current_hour < 21:
        return "18:00-20:59"
    else:  # 21-23
        return "21:00-23:59"

def find_nearest_node(lat: float, lon: float) -> str:
    """Find the nearest node in the graph to the given coordinates."""
    global node_coordinates
    
    min_distance = float('inf')
    nearest_node = None
    
    for node, (node_lat, node_lon) in node_coordinates.items():
        distance = math.sqrt((lat - node_lat)**2 + (lon - node_lon)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    
    # If the nearest node is too far (more than 10km), return None
    if min_distance > 0.1:  # Approximately 10km in degrees
        print(f"Warning: Nearest node is {min_distance:.6f} degrees away from input coordinates")
        return None
    
    print(f"Found nearest node at distance {min_distance:.6f} degrees")
    
    return nearest_node

def dijkstra_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, time_slot: str = "00:00-02:59") -> Tuple[List[str], float, float]:
    """
    Find the optimal route using Dijkstra's algorithm with custom cost function.
    Returns: (route_nodes, total_distance, total_pollution)
    """
    global road_graph, node_coordinates
    
    # Find nearest nodes to start and end points
    start_node = find_nearest_node(start_lat, start_lon)
    end_node = find_nearest_node(end_lat, end_lon)
    
    print(f"Start coordinates: ({start_lat}, {start_lon}) -> Node: {start_node}")
    print(f"End coordinates: ({end_lat}, {end_lon}) -> Node: {end_node}")
    
    if not start_node or not end_node:
        raise ValueError("Could not find nearest nodes to start or end points")
    
    if start_node == end_node:
        return [start_node], 0.0, 0.0
    
    # Define a weight function that calculates AQI cost dynamically with light penalties
    def dynamic_cost_weight(u, v, data):
        # Calculate midpoint of the edge
        lat1, lon1 = data['coordinates'][0][1], data['coordinates'][0][0]
        lat2, lon2 = data['coordinates'][1][1], data['coordinates'][1][0]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        # Calculate dynamic cost for this time slot using light penalties
        return calculate_road_cost_with_aqi(data['distance'], mid_lat, mid_lon, time_slot)
    
    # Run Dijkstra's algorithm with dynamic cost calculation
    try:
        shortest_path = nx.shortest_path(
            road_graph, 
            start_node, 
            end_node, 
            weight=dynamic_cost_weight
        )
    except nx.NetworkXNoPath:
        raise ValueError("No path found between the specified points. Try coordinates that are closer to major roads or use the example coordinates provided.")
    
    # Calculate total distance and pollution for the route
    total_distance = 0.0
    total_pollution = 0.0
    
    print(f"🔍 Route analysis for time slot '{time_slot}' - Path has {len(shortest_path)} nodes")
    
    for i in range(len(shortest_path) - 1):
        edge_data = road_graph[shortest_path[i]][shortest_path[i + 1]]
        total_distance += edge_data['distance']
        
        # Calculate AQI for this edge's midpoint using the specified time slot
        lat1, lon1 = edge_data['coordinates'][0][1], edge_data['coordinates'][0][0]
        lat2, lon2 = edge_data['coordinates'][1][1], edge_data['coordinates'][1][0]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        # Get AQI value for this point using the specified time slot
        aqi_value = interpolate_aqi_for_point(mid_lat, mid_lon, time_slot)
        
        # Debug: Print AQI and cost for each edge
        if i < 5 or i > len(shortest_path) - 6:  # Show first 5 and last 5 edges
            print(f"  Edge {i}: AQI={aqi_value:.0f}, Distance={edge_data['distance']:.3f}km")
        
        total_pollution += aqi_value * edge_data['distance']  # Weighted by distance
    
    print(f"📊 Total route: Distance={total_distance:.2f}km, Avg AQI={total_pollution/total_distance:.0f}")
    
    return shortest_path, total_distance, total_pollution

def dijkstra_route_distance_only(start_lat: float, start_lon: float, end_lat: float, end_lon: float, time_slot: str = "00:00-02:59") -> Tuple[List[str], float, float]:
    """
    Find the shortest route using Dijkstra's algorithm with distance-only cost function.
    Returns: (route_nodes, total_distance, total_pollution)
    """
    global road_graph, node_coordinates
    
    # Find nearest nodes to start and end points
    start_node = find_nearest_node(start_lat, start_lon)
    end_node = find_nearest_node(end_lat, end_lon)
    
    print(f"Distance-only route - Start: ({start_lat}, {start_lon}) -> Node: {start_node}")
    print(f"Distance-only route - End: ({end_lat}, {end_lon}) -> Node: {end_node}")
    
    if not start_node or not end_node:
        raise ValueError("Could not find nearest nodes to start or end points")
    
    if start_node == end_node:
        return [start_node], 0.0, 0.0
    
    # Run Dijkstra's algorithm using distance as weight directly
    try:
        shortest_path = nx.shortest_path(
            road_graph, 
            start_node, 
            end_node, 
            weight='distance'  # Use distance directly instead of cost
        )
    except nx.NetworkXNoPath:
        raise ValueError("No path found between the specified points.")
    
    # Calculate total distance and pollution for the route
    total_distance = 0.0
    total_pollution = 0.0
    
    print(f"🔍 Distance-only route analysis for time slot '{time_slot}' - Path has {len(shortest_path)} nodes")
    
    for i in range(len(shortest_path) - 1):
        edge_data = road_graph[shortest_path[i]][shortest_path[i + 1]]
        total_distance += edge_data['distance']
        
        # Calculate AQI for this edge's midpoint (for pollution calculation only)
        lat1, lon1 = edge_data['coordinates'][0][1], edge_data['coordinates'][0][0]
        lat2, lon2 = edge_data['coordinates'][1][1], edge_data['coordinates'][1][0]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        # Get AQI value for this point using the specified time slot (for reporting, not for routing)
        aqi_value = interpolate_aqi_for_point(mid_lat, mid_lon, time_slot)
        
        # Debug: Print distance and AQI for each edge
        if i < 5 or i > len(shortest_path) - 6:  # Show first 5 and last 5 edges
            print(f"  Distance-only Edge {i}: AQI={aqi_value:.0f}, Distance={edge_data['distance']:.3f}km")
        
        total_pollution += aqi_value * edge_data['distance']  # Weighted by distance
    
    print(f"📊 Distance-only route: Distance={total_distance:.2f}km, Avg AQI={total_pollution/total_distance:.0f}")
    
    return shortest_path, total_distance, total_pollution

def get_route_coordinates(route_nodes: List[str]) -> List[List[float]]:
    """Convert route nodes to a list of coordinate pairs for the frontend."""
    global road_graph, node_coordinates
    
    route_coordinates = []
    
    for i in range(len(route_nodes) - 1):
        current_node = route_nodes[i]
        next_node = route_nodes[i + 1]
        
        # Get the edge data
        edge_data = road_graph[current_node][next_node]
        
        # Add coordinates from the road segment
        for coord in edge_data['coordinates']:
            route_coordinates.append([coord[0], coord[1]])  # [lon, lat] format for Leaflet
    
    return route_coordinates

# Remove the startup event to prevent double initialization
# The graph is already built in the main section

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Air Quality Routing API",
        "status": "running",
        "endpoints": {
            "/route": "POST - Find optimal route between two points (with optional time_slot parameter)",
            "/route-auto-time": "POST - Find optimal route using current time automatically",
            "/route-distance": "POST - Find shortest route (distance only) with optional time_slot",
            "/stations": "GET - Get all stations (with optional time_slot parameter)",
            "/station/{name}/time-slots": "GET - Get all time slot data for a specific station",
            "/time-slots": "GET - Get all available time slots",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_nodes": road_graph.number_of_nodes() if road_graph else 0,
        "graph_edges": road_graph.number_of_edges() if road_graph else 0,
        "aqi_cache_size": len(aqi_cache) if aqi_cache else 0
    }

@app.post("/rebuild-graph")
async def rebuild_graph_endpoint():
    """Manually trigger graph rebuild (useful for development)."""
    global road_graph, road_data, node_coordinates, aqi_cache
    
    try:
        print("🔄 Manual graph rebuild requested...")
        
        # Load road data
        load_road_data()
        
        # Build graph
        build_graph()
        
        # Save to disk
        save_graph_to_disk()
        
        return {
            "status": "success",
            "message": "Graph rebuilt successfully",
            "nodes": road_graph.number_of_nodes() if road_graph else 0,
            "edges": road_graph.number_of_edges() if road_graph else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding graph: {str(e)}")

@app.get("/coverage-area")
async def get_coverage_area():
    """Get the bounding box of the road network coverage area."""
    global road_data
    
    if not road_data:
        raise HTTPException(status_code=500, detail="Road data not loaded")
    
    # Calculate bounding box from all road segments using ALL coordinates
    all_lats = []
    all_lons = []
    
    for road in road_data:
        coordinates = road['coordinates']
        for coord in coordinates:
            lon, lat = coord
            all_lats.append(lat)
            all_lons.append(lon)
    
    if not all_lats or not all_lons:
        raise HTTPException(status_code=500, detail="No valid coordinates found in road data")
    
    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lon = min(all_lons)
    max_lon = max(all_lons)
    
    # Add some padding to the bounding box
    lat_padding = (max_lat - min_lat) * 0.05
    lon_padding = (max_lon - min_lon) * 0.05
    
    return {
        "bounds": {
            "south": min_lat - lat_padding,
            "north": max_lat + lat_padding,
            "west": min_lon - lon_padding,
            "east": max_lon + lon_padding
        },
        "center": {
            "lat": (min_lat + max_lat) / 2,
            "lon": (min_lon + max_lon) / 2
        }
    }

@app.get("/road-segments")
async def get_road_segments():
    """Get all road segments for displaying on the map with interpolated AQI values."""
    global road_data
    
    if not road_data:
        raise HTTPException(status_code=500, detail="Road data not loaded")
    
    # Return all road segments with their coordinates and interpolated AQI
    segments = []
    for road in road_data:
        # Calculate interpolated AQI for the road segment midpoint
        mid_lat = (road['start_lat'] + road['end_lat']) / 2
        mid_lon = (road['start_lng'] + road['end_lng']) / 2
        interpolated_aqi = interpolate_aqi_for_point(mid_lat, mid_lon)
        
        segments.append({
            "coordinates": road['coordinates'],
            "road_name": road['road_name'],
            "aqi": interpolated_aqi,  # Use interpolated AQI instead of fake PM2.5
            "distance_km": road['distance_km']
        })
    
    return {"segments": segments}

@app.get("/stations")
async def get_stations(time_slot: str = None):
    """Get all air quality monitoring stations with their data for a specific time slot."""
    global aqi_data
    
    if aqi_data is None or aqi_data.empty:
        raise HTTPException(status_code=500, detail="AQI data not loaded")
    
    # If no time slot specified, use current time slot
    if time_slot is None:
        time_slot = get_current_time_slot()
    
    # Filter data for the specified time slot
    time_data = aqi_data[aqi_data['time_slot'] == time_slot].copy()
    
    if time_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for time slot: {time_slot}")
    
    # Get unique stations for this time slot
    stations = []
    seen_stations = set()
    
    for _, row in time_data.iterrows():
        station_name = row['station_name']
        if station_name not in seen_stations:
            seen_stations.add(station_name)
            
            stations.append({
                "station_name": station_name,
                "latitude": float(row['latitude']),
                "longitude": float(row['longitude']),
                "temperature": float(row['AT']) if pd.notna(row['AT']) else None,
                "humidity": float(row['RH']) if pd.notna(row['RH']) else None,
                "pm25": float(row['PM2_5']) if pd.notna(row['PM2_5']) else None,
                "pm10": float(row['PM10']) if pd.notna(row['PM10']) else None,
                "co2": float(row['CO2']) if pd.notna(row['CO2']) else None,
                "aqi": float(row['AQI']) if pd.notna(row['AQI']) else None,
                "time_slot": row['time_slot']
            })
    
    return {"stations": stations, "time_slot": time_slot}

@app.get("/station/{station_name}/time-slots")
async def get_station_time_slots(station_name: str):
    """Get all time slot data for a specific station."""
    global aqi_data
    
    if aqi_data is None or aqi_data.empty:
        raise HTTPException(status_code=500, detail="AQI data not loaded")
    
    # Filter data for this station
    station_data = aqi_data[aqi_data['station_name'] == station_name].copy()
    
    if station_data.empty:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
    
    # Get all time slots for this station
    time_slots = []
    for _, row in station_data.iterrows():
        time_slots.append({
            "time_slot": row['time_slot'],
            "aqi": float(row['AQI']) if pd.notna(row['AQI']) else None,
            "pm25": float(row['PM2_5']) if pd.notna(row['PM2_5']) else None,
            "pm10": float(row['PM10']) if pd.notna(row['PM10']) else None,
            "temperature": float(row['AT']) if pd.notna(row['AT']) else None,
            "humidity": float(row['RH']) if pd.notna(row['RH']) else None,
            "co2": float(row['CO2']) if pd.notna(row['CO2']) else None
        })
    
    # Sort by time slot
    time_slots.sort(key=lambda x: x['time_slot'])
    
    return {
        "station_name": station_name,
        "latitude": float(station_data.iloc[0]['latitude']),
        "longitude": float(station_data.iloc[0]['longitude']),
        "time_slots": time_slots
    }

@app.get("/time-slots")
async def get_available_time_slots():
    """Get all available time slots in the dataset."""
    global aqi_data
    
    if aqi_data is None or aqi_data.empty:
        raise HTTPException(status_code=500, detail="AQI data not loaded")
    
    # Get unique time slots
    time_slots = sorted(aqi_data['time_slot'].unique().tolist())
    
    return {
        "time_slots": time_slots,
        "current_time_slot": get_current_time_slot()
    }

@app.get("/debug-aqi/{lat}/{lon}")
async def debug_aqi_interpolation(lat: float, lon: float):
    """Debug AQI interpolation for a specific point."""
    global aqi_data
    
    if aqi_data is None or aqi_data.empty:
        raise HTTPException(status_code=500, detail="AQI data not loaded")
    
    # Get interpolated AQI
    aqi_value = interpolate_aqi_for_point(lat, lon)
    
    # Find nearby stations
    time_data = aqi_data[aqi_data['time_slot'] == "00:00-02:59"].copy()
    station_coords = time_data[['latitude', 'longitude']].values
    road_coords = np.array([[lat, lon]])
    distances = cdist(road_coords, station_coords, metric='euclidean') * 111  # Convert to km
    distances = distances[0]
    
    # Get stations within 10km
    nearby_stations = []
    for i, distance in enumerate(distances):
        if distance <= 10.0:
            station = time_data.iloc[i]
            nearby_stations.append({
                "station_name": station['station_name'],
                "latitude": float(station['latitude']),
                "longitude": float(station['longitude']),
                "aqi": float(station['AQI']),
                "distance_km": float(distance)
            })
    
    # Sort by distance
    nearby_stations.sort(key=lambda x: x['distance_km'])
    
    return {
        "point": {"lat": lat, "lon": lon},
        "interpolated_aqi": aqi_value,
        "nearby_stations": nearby_stations[:5]  # Top 5 closest
    }

@app.post("/route", response_model=RouteResponse)
async def find_route(request: RouteRequest):
    """
    Find the optimal route between two points considering air quality.
    
    The algorithm prioritizes routes with lower AQI (Air Quality Index) levels
    while still considering distance to find a reasonable balance.
    Uses interpolated AQI values from monitoring stations.
    """
    try:
        # Validate input coordinates
        if not (-90 <= request.start_lat <= 90 and -180 <= request.start_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid start coordinates")
        if not (-90 <= request.end_lat <= 90 and -180 <= request.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid end coordinates")
        
        # Use provided time slot or default
        time_slot = request.time_slot if request.time_slot else "00:00-02:59"
        
        # Find the optimal route
        route_nodes, total_distance, total_pollution = dijkstra_route(
            request.start_lat, 
            request.start_lon, 
            request.end_lat, 
            request.end_lon,
            time_slot
        )
        
        # Convert to coordinates for the frontend
        route_coordinates = get_route_coordinates(route_nodes)
        
        # Calculate average AQI for the route
        avg_aqi = total_pollution / total_distance if total_distance > 0 else 0
        
        # Determine AQI category
        if avg_aqi <= 50:
            category = "Good"
        elif avg_aqi <= 100:
            category = "Satisfactory"
        elif avg_aqi <= 200:
            category = "Moderate"
        elif avg_aqi <= 300:
            category = "Poor"
        elif avg_aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"
        
        message = f"Route found for {time_slot}! Distance: {total_distance:.2f} km, Average AQI: {avg_aqi:.0f} ({category})"
        
        return RouteResponse(
            route=route_coordinates,
            total_distance=total_distance,
            total_pollution=avg_aqi,
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error finding route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/route-auto-time", response_model=RouteResponse)
async def find_route_auto_time(request: RouteRequest):
    """
    Find the optimal route between two points using the current time slot automatically.
    
    This endpoint automatically determines the appropriate time slot based on the current time
    and finds the optimal route considering air quality for that specific time period.
    """
    try:
        # Validate input coordinates
        if not (-90 <= request.start_lat <= 90 and -180 <= request.start_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid start coordinates")
        if not (-90 <= request.end_lat <= 90 and -180 <= request.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid end coordinates")
        
        # Automatically determine the current time slot
        current_time_slot = get_current_time_slot()
        print(f"🕐 Auto-detected time slot: {current_time_slot}")
        
        # Find the optimal route
        route_nodes, total_distance, total_pollution = dijkstra_route(
            request.start_lat, 
            request.start_lon, 
            request.end_lat, 
            request.end_lon,
            current_time_slot
        )
        
        # Convert to coordinates for the frontend
        route_coordinates = get_route_coordinates(route_nodes)
        
        # Calculate average AQI for the route
        avg_aqi = total_pollution / total_distance if total_distance > 0 else 0
        
        # Determine AQI category
        if avg_aqi <= 50:
            category = "Good"
        elif avg_aqi <= 100:
            category = "Satisfactory"
        elif avg_aqi <= 200:
            category = "Moderate"
        elif avg_aqi <= 300:
            category = "Poor"
        elif avg_aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"
        
        message = f"Route found for current time ({current_time_slot})! Distance: {total_distance:.2f} km, Average AQI: {avg_aqi:.0f} ({category})"
        
        return RouteResponse(
            route=route_coordinates,
            total_distance=total_distance,
            total_pollution=avg_aqi,
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error finding route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/route-distance", response_model=RouteResponse)
async def find_route_distance_only(request: RouteRequest):
    """
    Find the shortest route between two points considering only distance (ignoring air quality).
    
    This algorithm finds the shortest path regardless of air quality conditions.
    Useful for comparison with AQI-optimized routes.
    """
    try:
        # Validate input coordinates
        if not (-90 <= request.start_lat <= 90 and -180 <= request.start_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid start coordinates")
        if not (-90 <= request.end_lat <= 90 and -180 <= request.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid end coordinates")
        
        # Use provided time slot or default
        time_slot = request.time_slot if request.time_slot else "00:00-02:59"
        
        # Find the shortest route (distance only)
        route_nodes, total_distance, total_pollution = dijkstra_route_distance_only(
            request.start_lat, 
            request.start_lon, 
            request.end_lat, 
            request.end_lon,
            time_slot
        )
        
        # Convert to coordinates for the frontend
        route_coordinates = get_route_coordinates(route_nodes)
        
        # Calculate average AQI for the route (for reporting only)
        avg_aqi = total_pollution / total_distance if total_distance > 0 else 0
        
        # Determine AQI category
        if avg_aqi <= 50:
            category = "Good"
        elif avg_aqi <= 100:
            category = "Satisfactory"
        elif avg_aqi <= 200:
            category = "Moderate"
        elif avg_aqi <= 300:
            category = "Poor"
        elif avg_aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"
        
        message = f"Shortest route found for {time_slot}! Distance: {total_distance:.2f} km, Average AQI: {avg_aqi:.0f} ({category})"
        
        return RouteResponse(
            route=route_coordinates,
            total_distance=total_distance,
            total_pollution=avg_aqi,
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error finding distance-only route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    print("Starting Clear Paths Backend with AQI Integration...")
    
    # Check if we can load the graph from disk
    if should_rebuild_graph():
        print("🔄 Building graph from source data...")
        
        # Load road data
        load_road_data()
        
        # Load AQI data
        if not load_aqi_data():
            print("⚠️ AQI data not available, using default values")
        
        # Build graph
        build_graph()
        
        # Save graph to disk for future use
        save_graph_to_disk()
        
    else:
        print("📂 Loading graph from disk...")
        
        # Load AQI data (needed for interpolation)
        if not load_aqi_data():
            print("⚠️ AQI data not available, using default values")
        
        # Load graph from disk
        if not load_graph_from_disk():
            print("❌ Failed to load graph from disk, rebuilding...")
            load_road_data()
            build_graph()
            save_graph_to_disk()
    
    print("✅ Backend ready! Starting server...")
    
    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 