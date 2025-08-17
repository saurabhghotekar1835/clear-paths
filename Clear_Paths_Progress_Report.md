# **📋 Clear Paths - Milestone 1 Progress Report**

### **🏆 Project Overview**

**Project Name:** Clear Paths - AI-Powered Air Quality Routing System  
**Team:** [Your Team Name]  
**Submission Date:** 12 August 2025  
**Problem Statement:** Air pollution significantly impacts public health and reduces tourism, with polluted areas experiencing a drastic drop in return visits. Current travel planning tools often prioritize shortest distance or fastest time, neglecting air quality as a critical factor.

---

## **🎯 Solution Overview**

**Clear Paths** is an AI-powered web application that helps users plan their daily commutes and eco-tourism trips by prioritizing routes with lower air pollution exposure. The system integrates real-time air quality data with geospatial information to provide optimized paths, identify clean-air destinations, and offer personalized routing based on current time-specific pollution levels.

---

## **📊 Technical Implementation**

### **🔄 Data Processing Pipeline**

#### **📈 Raw Data Sources**

**Air Quality Data:**
- **Source:** 75 monitoring stations across Gurugram
- **Time Period:** November 2024 - July 2025 (9 months)
- **Parameters:** PM2.5, PM10, Temperature (AT), Humidity (RH), CO2
- **Raw Data Points:** 9 months × 75 stations × 24 hours × 30 days = ~4.86M data points

**Road Network Data:**
- **Primary Source:** ArcGIS Living Atlas - Highways and Roads
- **Secondary Source:** OpenStreetMap (OSM) PBF data
- **Processing Tool:** Custom Python script (`extract_real_roads.py`)
- **Output Format:** GeoJSON with comprehensive road attributes

#### **🔧 Data Processing Steps**

**Step 1: Data Selection & Validation**
- **Selected Dataset:** July 2025 (most complete with 30 daily readings per station)
- **Data Quality:** Validated for missing readings and inconsistencies
- **Coverage:** 75 stations with complete environmental parameters

**Step 2: Hourly Aggregation**
- **Input:** 30 readings per station per hour
- **Process:** Statistical averaging with outlier removal
- **Output:** 1 average reading per hour (24 hours total)

**Step 3: Time Slot Creation**
- **Rationale:** Handle missing hourly data and create manageable time periods
- **Slots:** 8 time periods of 3 hours each
  - `00:00-02:59` (Midnight to 3 AM)
  - `03:00-05:59` (3 AM to 6 AM)
  - `06:00-08:59` (6 AM to 9 AM) ← **Morning Rush**
  - `09:00-11:59` (9 AM to 12 PM)
  - `12:00-14:59` (12 PM to 3 PM)
  - `15:00-17:59` (3 PM to 6 PM)
  - `18:00-20:59` (6 PM to 9 PM) ← **Evening Rush**
  - `21:00-23:59` (9 PM to 12 AM)

**Step 4: AQI Calculation**
- **Formula:** Indian AQI with PM2.5/PM10 breakpoints
- **Method:** Linear interpolation for sub-indices
- **Output:** 600 clean, time-slot specific data points (75 stations × 8 slots)

#### **📊 Final Processed Data Statistics**

**Air Quality Dataset (`gurugram_air_quality_with_aqi.csv`):**
- **Total Readings:** 508 (75 stations × 8 time slots, some missing)
- **Unique Stations:** 75 monitoring stations
- **Time Slots:** 8 (3-hour periods)
- **Geographic Coverage:** 599.4 km²
- **Latitude Range:** 28.335762° to 28.532499°
- **Longitude Range:** 76.898974° to 77.146263°
- **AQI Range:** 19.0 (Good) to 271.0 (Poor)
- **Average AQI:** 103.4 (Moderate)

**Time Slot Distribution:**
- `00:00-02:59`: 64 readings
- `03:00-05:59`: 64 readings
- `06:00-08:59`: 63 readings
- `09:00-11:59`: 73 readings
- `12:00-14:59`: 60 readings
- `15:00-17:59`: 62 readings
- `18:00-20:59`: 61 readings
- `21:00-23:59`: 61 readings

### **🛣️ Road Network Implementation**

#### **🗺️ Map Base Layer**
- **Technology:** OpenStreetMap (OSM) tiles
- **Implementation:** Leaflet.js tile layer
- **URL:** `https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`
- **Attribution:** OpenStreetMap contributors

#### **🛣️ Road Network Data**

**Source & Processing:**
- **Primary Source:** ArcGIS Living Atlas - Highways and Roads
- **Processing Script:** `extract_real_roads.py`
- **Output File:** `gurugram_roads_real.geojson`
- **File Size:** ~45MB

**Road Network Statistics:**
- **Total Road Segments:** 45,215 segments
- **Geographic Coverage:** 739.2 km²
- **Bounding Box:**
  - **Min Latitude:** 28.350003°
  - **Max Latitude:** 28.549998°
  - **Min Longitude:** 76.850001°
  - **Max Longitude:** 77.149997°

**Road Type Distribution:**
- **Residential:** 28,093 segments (62.1%)
- **Service:** 10,535 segments (23.3%)
- **Tertiary:** 1,231 segments (2.7%)
- **Track:** 921 segments (2.0%)
- **Living Street:** 892 segments (2.0%)
- **Secondary:** 778 segments (1.7%)
- **Footway:** 660 segments (1.5%)
- **Secondary Link:** 325 segments (0.7%)
- **Unclassified:** 268 segments (0.6%)
- **Trunk:** 238 segments (0.5%)

#### **🕸️ Graph Construction**

**Graph Statistics:**
- **Total Nodes:** 253,291 unique coordinate points
- **Total Edges:** 275,558 road segments
- **Node Deduplication:** 275,558 coordinates → 253,291 unique nodes
- **Graph Type:** Directed weighted graph with custom cost function

**Graph Building Process:**
1. **Coordinate Extraction:** All coordinate points from road segments
2. **Node Creation:** Unique nodes for each coordinate point
3. **Edge Creation:** Connections between consecutive coordinates
4. **Distance Calculation:** Haversine formula for accurate distances
5. **Cost Assignment:** Dynamic cost based on distance and interpolated AQI

### **⚡ Core Algorithm**

#### **🎯 Modified Dijkstra's Algorithm**

**Base Algorithm:** NetworkX shortest path with custom weight function  
**Innovation:** Dynamic cost calculation without graph rebuilding

**Cost Function:** `cost = distance_km × aqi_weight`

**AQI Weighting System:**
- **Good (0-50):** 1.0x (no penalty)
- **Satisfactory (51-100):** 2.0x (small penalty)
- **Moderate (101-200):** 25.0x (extreme penalty)
- **Poor (201-300):** 200.0x (maximum penalty)
- **Very Poor (301-400):** 500.0x (extreme penalty)
- **Severe (>400):** 1000.0x (maximum penalty)

#### **🌬️ AQI Integration**

**Interpolation Method:** Inverse Distance Weighting (IDW)
- **Influence Radius:** 5km (reduced from 10km for localization)
- **Maximum Stations:** 5 nearest stations per road point
- **Weighting Formula:** `weight = 1 / (distance²)`

**Caching System:**
- **Cache Type:** Time slot specific AQI interpolation results
- **Performance:** Instant time slot switching (<1 second)
- **Memory Efficiency:** Single road network structure

### **🌐 Backend Architecture**

#### **🛠️ Technology Stack**

**Core Framework:**
- **FastAPI:** Modern Python web framework
- **Uvicorn:** ASGI server for high performance
- **Pydantic:** Data validation and serialization

**Data Processing:**
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **SciPy:** Scientific computing and spatial analysis

**Geospatial Processing:**
- **GeoPandas:** Geospatial data manipulation
- **Shapely:** Geometric operations
- **Fiona:** Geospatial data I/O

**Graph Algorithms:**
- **NetworkX:** Graph theory and algorithms
- **Custom Weight Functions:** Dynamic cost calculation

#### **🔌 API Endpoints**

**Route Finding:**
- `POST /route` - Find optimal route with time slot parameter
- `POST /route-auto-time` - Auto-detect current time for routing
- `POST /route-distance` - Shortest distance route comparison

**Data Access:**
- `GET /stations` - Time slot specific station data
- `GET /time-slots` - Available time slots
- `GET /station/{name}/time-slots` - Detailed station data
- `GET /coverage-area` - Road network bounding box
- `GET /road-segments` - All road segments with AQI

**System:**
- `GET /` - API documentation
- `GET /health` - System health check
- `POST /rebuild-graph` - Graph reconstruction

#### **📊 Performance Metrics**

**Route Calculation:**
- **Average Time:** ~9 seconds for complex routes
- **Time Slot Switching:** <1 second (instant)
- **API Response Time:** <500ms for station data
- **Memory Usage:** Efficient caching system

**System Performance:**
- **Graph Loading:** ~30 seconds (cached after first load)
- **Station Data:** <2 seconds for all 75 stations
- **Map Rendering:** <3 seconds for road network 