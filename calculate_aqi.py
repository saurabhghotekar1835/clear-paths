#!/usr/bin/env python3
"""
Calculate Indian AQI from PM2.5 and PM10 values
Uses Indian AQI breakpoints and linear interpolation formula
"""

import pandas as pd
import numpy as np

def calculate_aqi(pm25_value, pm10_value):
    """
    Calculate Indian AQI using PM2.5 and PM10 values.
    Returns the maximum of the two sub-indices.
    
    Args:
        pm25_value (float): PM2.5 concentration in µg/m³
        pm10_value (float): PM10 concentration in µg/m³
    
    Returns:
        float: AQI value (0-500)
    """
    
    # Indian AQI breakpoints for PM2.5 and PM10
    pm25_breakpoints = [
        (0, 30, 0, 50),      # AQI 0-50
        (31, 60, 51, 100),   # AQI 51-100
        (61, 90, 101, 200),  # AQI 101-200
        (91, 120, 201, 300), # AQI 201-300
        (121, 250, 301, 400), # AQI 301-400
        (251, 380, 401, 500)  # AQI 401-500
    ]
    
    pm10_breakpoints = [
        (0, 50, 0, 50),      # AQI 0-50
        (51, 100, 51, 100),  # AQI 51-100
        (101, 250, 101, 200), # AQI 101-200
        (251, 350, 201, 300), # AQI 201-300
        (351, 430, 301, 400), # AQI 301-400
        (431, 1000, 401, 500) # AQI 401-500
    ]
    
    def calculate_sub_index(concentration, breakpoints):
        """Calculate sub-index for a given concentration using breakpoints."""
        
        # Handle edge cases
        if concentration <= 0:
            return 0
        if concentration > breakpoints[-1][1]:  # Above highest breakpoint
            return breakpoints[-1][3]  # Return maximum AQI
        
        # Find the appropriate breakpoint range
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Linear interpolation formula
                # Ip = [(I_Hi - I_Lo) / (BP_Hi - BP_Lo)] * (C_p - BP_Lo) + I_Lo
                sub_index = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return round(sub_index)
        
        return 0  # Fallback
    
    # Calculate sub-indices for PM2.5 and PM10
    pm25_sub_index = calculate_sub_index(pm25_value, pm25_breakpoints)
    pm10_sub_index = calculate_sub_index(pm10_value, pm10_breakpoints)
    
    # Return the maximum of the two sub-indices
    return max(pm25_sub_index, pm10_sub_index)

def process_csv_file():
    """Process the CSV file and add AQI column."""
    
    input_file = "gurugram_air_quality_3hour_slots.csv"
    output_file = "gurugram_air_quality_with_aqi.csv"
    
    try:
        # Read the CSV file
        print(f"📖 Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"📊 Original data shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check if required columns exist
        if 'PM2_5' not in df.columns or 'PM10' not in df.columns:
            print("❌ Error: PM2_5 or PM10 columns not found!")
            return False
        
        # Calculate AQI for each row
        print("🧮 Calculating AQI values...")
        df['AQI'] = df.apply(lambda row: calculate_aqi(row['PM2_5'], row['PM10']), axis=1)
        
        # Show AQI statistics
        print(f"\n📊 AQI Statistics:")
        print(f"   Minimum AQI: {df['AQI'].min()}")
        print(f"   Maximum AQI: {df['AQI'].max()}")
        print(f"   Average AQI: {df['AQI'].mean():.1f}")
        print(f"   Median AQI: {df['AQI'].median():.1f}")
        
        # Show AQI distribution
        print(f"\n📈 AQI Distribution:")
        aqi_ranges = [
            (0, 50, "Good"),
            (51, 100, "Satisfactory"),
            (101, 200, "Moderate"),
            (201, 300, "Poor"),
            (301, 400, "Very Poor"),
            (401, 500, "Severe")
        ]
        
        for low, high, category in aqi_ranges:
            count = len(df[(df['AQI'] >= low) & (df['AQI'] <= high)])
            percentage = (count / len(df)) * 100
            print(f"   {category} ({low}-{high}): {count} readings ({percentage:.1f}%)")
        
        # Show sample data
        print(f"\n📋 Sample data with AQI:")
        sample_cols = ['station_name', 'time_slot', 'PM2_5', 'PM10', 'AQI']
        print(df[sample_cols].head(10))
        
        # Save the updated CSV
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved updated data to: {output_file}")
        print(f"📁 File size: {len(df)} rows, {len(df.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def test_aqi_calculation():
    """Test the AQI calculation with sample values."""
    
    print("🧪 Testing AQI calculation with sample values:")
    print("=" * 50)
    
    test_cases = [
        (15, 30, "Good air quality"),
        (45, 80, "Satisfactory air quality"),
        (75, 150, "Moderate air quality"),
        (110, 280, "Poor air quality"),
        (200, 400, "Very poor air quality"),
        (300, 500, "Severe air quality")
    ]
    
    for pm25, pm10, description in test_cases:
        aqi = calculate_aqi(pm25, pm10)
        print(f"PM2.5: {pm25} µg/m³, PM10: {pm10} µg/m³ → AQI: {aqi} ({description})")

if __name__ == "__main__":
    print("🌬️ Indian AQI Calculator")
    print("=" * 40)
    
    # Test the calculation first
    test_aqi_calculation()
    print()
    
    # Process the CSV file
    if process_csv_file():
        print("\n✅ AQI calculation completed successfully!")
    else:
        print("\n❌ AQI calculation failed!") 