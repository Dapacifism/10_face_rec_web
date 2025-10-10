import os
from imutils import paths
import face_recognition
import pickle
import cv2

def train_model():
    print("[INFO] start processing faces...")
    imagePaths = list(paths.list_images("dataset"))
    
    if len(imagePaths) == 0:
        print("[ERROR] No images found in dataset folder")
        return False
    
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]
        
        image = cv2.imread(imagePath)
        if image is None:
            continue
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    if len(knownEncodings) == 0:
        print("[ERROR] No face encodings found")
        return False

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))

    print(f"[INFO] Training complete. {len(knownEncodings)} encodings saved to 'encodings.pickle'")
    return True

# This allows the file to be run directly
if __name__ == "__main__":
    train_model()