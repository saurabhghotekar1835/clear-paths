# 🌬️ Air Quality Routing Application

A full-stack web application that finds optimal routes through Gurugram, India, while minimizing exposure to air pollution. The application uses a custom routing algorithm that heavily penalizes high-AQI areas to prioritize cleaner air routes based on interpolated AQI values from monitoring stations.

## 🏗️ Architecture

### Backend (Python FastAPI)
- **Data Ingestion**: Reads road network from GeoJSON and AQI data from CSV monitoring stations
- **Graph Construction**: Builds a NetworkX graph from road segments with interpolated AQI data
- **Routing Algorithm**: Implements Dijkstra's algorithm with AQI-weighted cost function
- **API Endpoints**: RESTful API for route finding

### Frontend (HTML/JavaScript)
- **Interactive Map**: Leaflet.js for map visualization
- **User Interface**: Bootstrap-styled form for coordinate input
- **Real-time Routing**: JavaScript fetches routes from the backend API

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Backend Server**
   ```bash
   python backend.py
   ```
   The server will start on `http://localhost:8000`

3. **Open the Frontend**
   - Open `index.html` in your web browser
   - Or serve it using a simple HTTP server:
     ```bash
     # Python 3
     python -m http.server 8080
     # Then open http://localhost:8080
     ```

## 📖 Usage Guide

### Using the Application

1. **Enter Coordinates**: 
   - Fill in the start and end coordinates (latitude/longitude)
   - Or use the example coordinate buttons for quick testing

2. **Find Route**: 
   - Click "Find Optimal Route" to calculate the cleanest path
   - The algorithm will prioritize routes with lower AQI levels

3. **View Results**:
   - The optimal route will be displayed on the map
   - Route information shows distance and average AQI levels

### Example Coordinates (Gurugram Area)

The application includes three example coordinate pairs for testing:

- **Example 1**: Gurugram center to Sohna Road area
- **Example 2**: Sector 56 area to central Gurugram  
- **Example 3**: Sohna Road south to central area

## 🔧 Technical Details

### Custom Cost Function

The routing algorithm uses a custom cost function that combines distance and air quality with a strong AQI penalty:

```python
cost = distance_km * aqi_weight
```

Where `aqi_weight` is determined by AQI category:
- **Good (≤50)**: 1.0x (no penalty)
- **Satisfactory (51-100)**: 2.0x (very small penalty)
- **Moderate (101-200)**: 50.0x (extreme penalty)
- **Poor (201-300)**: 200.0x (maximum penalty)
- **Very Poor (301-400)**: 500.0x (extreme penalty)
- **Severe (>400)**: 1000.0x (maximum penalty)

This ensures routes absolutely avoid high-pollution areas, with extreme penalties for orange areas to force preference for green/yellow routes.

### API Endpoints

- `GET /`: API information and status
- `GET /health`: Health check with graph statistics
- `POST /route`: Find optimal route between two points

### Request Format
```json
{
  "start_lat": 28.4595,
  "start_lon": 77.0266,
  "end_lat": 28.4086,
  "end_lon": 77.0429
}
```

### Response Format
```json
{
  "route": [[lon1, lat1], [lon2, lat2], ...],
  "total_distance": 2.5,
  "total_pollution": 65.3,
  "message": "Route found! Distance: 2.50 km, Average AQI: 65"
}
```

## 🗂️ File Structure

```
clear skies/
├── backend.py                                    # FastAPI backend server
├── index.html                                    # Frontend web interface
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
└── gurugram_road_segments_with_air_quality.geojson  # Road data with air quality
```

## 🔍 Data Format

The GeoJSON file contains road segments with the following properties:
- `road_id`: Unique identifier for each road segment
- `road_name`: Name of the road
- `road_type`: Type of road (trunk, secondary, etc.)
- `distance_km`: Length of the road segment in kilometers
- `start_lat/lng`, `end_lat/lng`: Start and end coordinates
- `geometry`: LineString coordinates of the road segment

The CSV file contains monitoring station data with:
- `station_name`: Name of the monitoring station
- `latitude`, `longitude`: Station coordinates
- `PM2_5`, `PM10`: Raw pollution measurements
- `AQI`: Calculated Air Quality Index
- `time_slot`: Time period for the measurement

## 🌐 Air Quality Categories

The application uses standard AQI categories:
- **Good (0-50)**: Green
- **Satisfactory (51-100)**: Yellow  
- **Moderate (101-200)**: Orange
- **Poor (201-300)**: Red
- **Very Poor (301-400)**: Dark Red
- **Severe (401+)**: Purple

## 🛠️ Development

### Running in Development Mode

The backend includes auto-reload for development:
```bash
python backend.py
```

### API Documentation

Once the server is running, visit:
- `http://localhost:8000/docs` - Interactive API documentation (Swagger UI)
- `http://localhost:8000/redoc` - Alternative API documentation

### Troubleshooting

1. **Port Already in Use**: Change the port in `backend.py` line 280
2. **CORS Issues**: The backend includes CORS middleware for local development
3. **Data Loading Errors**: Ensure the GeoJSON file is in the same directory as `backend.py`

## 📊 Performance

- **First Startup**: ~30-60 seconds (builds graph from scratch)
- **Subsequent Starts**: ~2-5 seconds (loads cached graph from disk)
- **Route Calculation**: Typically <1 second for most routes
- **Memory Usage**: ~100-200MB depending on data size

### Graph Caching

The backend automatically caches the built graph to disk (`road_graph_cache.pkl`) and only rebuilds it when:
- Source data files (GeoJSON/CSV) are modified
- Cache files are missing or corrupted
- Manual rebuild is triggered via `/rebuild-graph` endpoint

This dramatically improves startup time after the initial build.

## 🔒 Security Notes

- CORS is configured to allow all origins (`*`) for development
- In production, specify exact frontend domains
- Input validation is implemented for coordinate ranges

## 📝 License

This project is for educational and research purposes. The air quality data and routing algorithm are designed to help users make informed decisions about their travel routes based on environmental factors.

## 🤝 Contributing

Feel free to improve the application by:
- Optimizing the routing algorithm
- Adding more air quality metrics
- Enhancing the user interface
- Adding real-time air quality data integration 