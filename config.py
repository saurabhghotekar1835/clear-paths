import os
from typing import List

class Settings:
    # Environment
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"
    ).split(",") if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]
    
    # File Paths
    GRAPH_CACHE_FILE: str = os.getenv("GRAPH_CACHE_FILE", "road_graph_cache.pkl")
    GRAPH_METADATA_FILE: str = os.getenv("GRAPH_METADATA_FILE", "graph_metadata.json")
    
    # Data Files
    AQI_DATA_FILE: str = os.getenv("AQI_DATA_FILE", "aqi_data.csv")
    ROAD_DATA_FILE: str = os.getenv("ROAD_DATA_FILE", "road_network.geojson")

# Create settings instance
settings = Settings() 