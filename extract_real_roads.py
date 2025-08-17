#!/usr/bin/env python3
"""
Extract ALL real roads from OSM PBF data
"""

import os
import subprocess
import json
import xml.etree.ElementTree as ET

def convert_osm_to_xml():
    """Convert OSM PBF to XML format first."""
    
    input_file = "gurugram.osm.pbf"
    xml_file = "gurugram.osm"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' not found!")
        return False
    
    print(f"🔄 Converting PBF to XML format...")
    
    # Use osmosis to convert PBF to XML
    osmosis_lib = "osmosis/lib/default"
    jar_files = []
    for file in os.listdir(osmosis_lib):
        if file.endswith('.jar'):
            jar_files.append(os.path.join(osmosis_lib, file))
    
    classpath = os.pathsep.join(jar_files)
    
    cmd = [
        "java", "-cp", classpath,
        "org.openstreetmap.osmosis.core.Osmosis",
        "--read-pbf", input_file,
        "--write-xml", xml_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(xml_file):
            size_mb = os.path.getsize(xml_file) / (1024*1024)
            print(f"✅ Converted to XML: {xml_file}")
            print(f"📁 XML file size: {size_mb:.1f} MB")
            return True
        else:
            print(f"❌ XML conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ XML conversion error: {e}")
        return False

def extract_roads_from_xml():
    """Extract all road features from XML and convert to GeoJSON."""
    
    xml_file = "gurugram.osm"
    output_file = "gurugram_roads_real.geojson"
    
    if not os.path.exists(xml_file):
        print(f"❌ XML file '{xml_file}' not found!")
        return False
    
    print(f"🔄 Extracting roads from XML...")
    
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract nodes (coordinates)
        nodes = {}
        for node in root.findall('.//node'):
            node_id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            nodes[node_id] = [lon, lat]  # GeoJSON format: [lon, lat]
        
        print(f"📍 Found {len(nodes)} nodes")
        
        # Extract ways (roads)
        road_features = []
        road_count = 0
        
        for way in root.findall('.//way'):
            # Check if this way is a road
            is_road = False
            road_name = "Unnamed Road"
            highway_type = "unknown"
            
            for tag in way.findall('tag'):
                k = tag.get('k')
                v = tag.get('v')
                
                if k == 'highway':
                    is_road = True
                    highway_type = v
                elif k == 'name':
                    road_name = v
            
            if is_road:
                # Get coordinates for this road
                coordinates = []
                for nd in way.findall('nd'):
                    node_id = nd.get('ref')
                    if node_id in nodes:
                        coordinates.append(nodes[node_id])
                
                if len(coordinates) >= 2:  # Need at least 2 points for a line
                    # Calculate approximate distance
                    distance_km = 0
                    for i in range(len(coordinates) - 1):
                        lat1, lon1 = coordinates[i][1], coordinates[i][0]
                        lat2, lon2 = coordinates[i+1][1], coordinates[i+1][0]
                        # Simple distance calculation
                        distance_km += ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111
                    
                    road_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coordinates
                        },
                        "properties": {
                            "road_name": road_name,
                            "highway": highway_type,
                            "distance_km": round(distance_km, 2),
                            "pm25": 50.0  # Default PM2.5 value
                        }
                    })
                    road_count += 1
                    
                    if road_count % 100 == 0:
                        print(f"   Processed {road_count} roads...")
        
        # Create GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": road_features
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        size_mb = os.path.getsize(output_file) / (1024*1024)
        print(f"✅ Extracted roads to GeoJSON: {output_file}")
        print(f"📁 GeoJSON file size: {size_mb:.1f} MB")
        print(f"🛣️ Found {len(road_features)} road features")
        
        # Show some statistics
        highway_types = {}
        for feature in road_features:
            hw_type = feature['properties']['highway']
            highway_types[hw_type] = highway_types.get(hw_type, 0) + 1
        
        print(f"\n📊 Road Types:")
        for hw_type, count in sorted(highway_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {hw_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Road extraction failed: {e}")
        return False

def main():
    """Main function."""
    print("🗺️ Extract ALL Real Roads from OSM Data")
    print("=" * 50)
    
    # Convert PBF to XML
    if not convert_osm_to_xml():
        return
    
    # Extract roads from XML
    if not extract_roads_from_xml():
        return
    
    print("\n✅ Road extraction complete!")
    print("📁 Files created:")
    print("   - gurugram.osm (XML format)")
    print("   - gurugram_roads_real.geojson (ALL roads)")
    
    print("\n💡 Next steps:")
    print("1. Update your backend to use gurugram_roads_real.geojson")
    print("2. Test the road network visualization")
    print("3. You should see ALL roads in Gurugram!")

if __name__ == "__main__":
    main() 