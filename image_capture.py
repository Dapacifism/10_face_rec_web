import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time
import sys

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name, photo_count=10):
    try:
        folder = create_folder(name)
        
        # Initialize the camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()

        # Allow camera to warm up
        time.sleep(2)

        captured_count = 0
        
        print(f"Taking {photo_count} photos for {name}...")
        
        for i in range(photo_count):
            # Capture frame from Pi Camera
            frame = picam2.capture_array()
            
            # Save photo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            captured_count += 1
            print(f"Photo {captured_count}/{photo_count} saved: {filename}")
            
            time.sleep(0.5)  # Wait between captures
    
        # Clean up
        picam2.stop()
        print(f"Photo capture completed. {captured_count} photos saved for {name}.")
        return True
        
    except Exception as e:
        print(f"Error in capture_photos: {e}")
        return False

if __name__ == "__main__":
    # For standalone testing
    if len(sys.argv) > 1:
        person_name = sys.argv[1]
        capture_photos(person_name, 10)
    else:
        # Default name for testing
        capture_photos("TestUser", 5)