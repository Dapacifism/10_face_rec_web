import os
from imutils import paths
import face_recognition
import pickle
import cv2

def train_model():
    try:
        print("[INFO] start processing faces...")
        imagePaths = list(paths.list_images("dataset"))
        
        if len(imagePaths) == 0:
            print("[INFO] No images found in dataset folder - creating empty encodings")
            data = {"encodings": [], "names": []}
            with open("encodings.pickle", "wb") as f:
                f.write(pickle.dumps(data))
            return True
        
        knownEncodings = []
        knownNames = []

        for (i, imagePath) in enumerate(imagePaths):
            print(f"[INFO] processing image {i + 1}/{len(imagePaths)}: {imagePath}")
            name = imagePath.split(os.path.sep)[-2]
            
            try:
                image = cv2.imread(imagePath)
                if image is None:
                    print(f"[WARNING] Could not load image: {imagePath}")
                    continue
                
                # Skip images that are too small
                if image.shape[0] < 50 or image.shape[1] < 50:
                    print(f"[WARNING] Image too small: {imagePath}")
                    continue
                
                # Skip images that are too large (can cause memory issues)
                if image.shape[0] > 2000 or image.shape[1] > 2000:
                    print(f"[INFO] Resizing large image: {imagePath}")
                    image = cv2.resize(image, (1000, 1000))
                
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Try with different models
                try:
                    boxes = face_recognition.face_locations(rgb, model="hog")
                except:
                    print(f"[INFO] Trying CNN model for {imagePath}")
                    boxes = face_recognition.face_locations(rgb, model="cnn")
                
                if len(boxes) == 0:
                    print(f"[WARNING] No faces detected in: {imagePath}")
                    continue
                    
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {imagePath}: {e}")
                continue

            if len(knownEncodings) == 0:
                print("[WARNING] No face encodings found - creating empty encodings")
                # Create empty encodings file
                data = {"encodings": [], "names": []}
                with open("encodings.pickle", "wb") as f:
                    f.write(pickle.dumps(data))
                return True

        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        with open("encodings.pickle", "wb") as f:
            f.write(pickle.dumps(data))

            print(f"[INFO] Training complete. {len(knownEncodings)} encodings saved to 'encodings.pickle'")
            return True
        
    except Exception as e:
        print(f"Error in model training: {e}")
    return False

# This allows the file to be run directly
if __name__ == "__main__":
    train_model()