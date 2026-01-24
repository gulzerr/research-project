import pandas as pd
import requests
import urllib3

# Disable SSL warnings for university server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    # Read the parquet file
    df = pd.read_parquet('data/fade_results_complete.parquet')
    print(df.head())
    print("Hello from research-project!")

# Configuration
base_url = ""

# You need to provide your credentials
USERNAME = ""  # Replace with your university username
PASSWORD = ""  # Replace with your university password // 6JlludZk - instance

def login():
    """Get a valid session ID"""
    params = {
        "api": "SYNO.API.Auth",
        "version": "3",
        "method": "login",
        "account": USERNAME,
        "passwd": PASSWORD,
        "session": "FileStation",
        "format": "sid"
    }
    
    try:
        response = requests.get(base_url, params=params, verify=False)
        data = response.json()
        
        if data.get("success"):
            sid = data["data"]["sid"]
            print(f"✓ Login successful. Session ID: {sid[:10]}...")
            return sid
        else:
            print(f"Login failed: {data}")
            return None
    except Exception as e:
        print(f"Login error: {e}")
        return None