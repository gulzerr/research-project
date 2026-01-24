import requests
from PIL import Image
from io import BytesIO
import urllib3

# Disable SSL warnings for university server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_url = ""
folder_path = ""

def process_image(image_id, sid):
    file_path = f"{folder_path}sa_{image_id}.jpg"
    
    params = {
        "api": "SYNO.FileStation.Download",
        "version": "2",
        "method": "download",
        "path": file_path,
        "mode": "download",
        "_sid": sid
    }

    try:
        # Stream the image into memory (disable SSL verification for university server)
        response = requests.get(base_url, params=params, stream=True, verify=False)
        
        print(f"Status code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"Content length: {len(response.content)} bytes")
        
        if response.status_code == 200:            
            # ...existing code...
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content))
                    print(f"✓ Processing sa_{image_id}.jpg: {img.size} {img.format}")
                    # img.show()  # Show the image
                    # img.close()
                    return img
                except Exception as img_err:
                    print(f"Failed to open image: {img_err}")
                    print(f"Response (first 500 bytes): {response.content[:500]}")
            else:
                print(f"Failed to access image {image_id}: {response.status_code}")
                print(f"Response: {response.text[:500]}")
            # ...existing code...
        else:
            print(f"Failed to access image {image_id}: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

