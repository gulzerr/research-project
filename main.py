# Login first to get session ID
from src.login import login
from src.fetch_image import process_image

sid = login()

if sid:
    # Example: Process first 2 images
    for i in range(1, 3):
        img = process_image(i, sid)
        img.show()  # Show the image
        img.close()
else:
    print("Failed to login. Cannot process images.")

