from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import os
import threading
import time
from datetime import datetime
import pickle
import face_recognition
import numpy as np
from picamera2 import Picamera2
import shutil
import lgpio


app = Flask(__name__)

# GPIO setup for buzzer
BUZZER_PIN = 17  # You can change this to your actual pin (BCM numbering)
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, BUZZER_PIN)

# Global variables
recognition_active = False
current_frame = None
frame_lock = threading.Lock()
known_face_encodings = []
known_face_names = []
unknown_face_counter = 0
unknown_face_encodings = []  # Store encodings of detected unknown faces
last_unknown_check = 0  # Time of last unknown face detection
current_detections = {"known": [], "unknown": 0}  # Track current frame detections

def buzz(duration=0.5):
    """Activate buzzer for a short duration"""
    try:
        lgpio.gpio_write(h, BUZZER_PIN, 1)
        time.sleep(duration)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
    except Exception as e:
        print(f"[BUZZER ERROR] {e}")


# Initialize camera
def init_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)}
        ))
        picam2.start()
        time.sleep(2)  # Camera warm-up
        return picam2
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return None

camera = init_camera()

def load_encodings():
    global known_face_encodings, known_face_names
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        print(f"[INFO] Loaded {len(known_face_names)} face encodings")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load encodings: {e}")
        return False

# Load encodings on startup
load_encodings()

def generate_frames():
    global recognition_active, current_frame, camera, unknown_face_counter
    
    if camera is None:
        camera = init_camera()
        if camera is None:
            return
    
    while recognition_active:
        try:
            frame = camera.capture_array()
            
            if recognition_active:
                # Perform face recognition
                frame = process_frame_for_recognition(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            with frame_lock:
                current_frame = frame_bytes
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(1)

def process_frame_for_recognition(frame):
    global known_face_encodings, known_face_names, unknown_face_counter, unknown_face_encodings, last_unknown_check, current_detections
    
    # Reset current detections for this frame
    current_detections = {"known": [], "unknown": 0}
    
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    current_time = time.time()
    face_names = []
    for face_encoding in face_encodings:
        # Check if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name not in current_detections["known"]:
                    current_detections["known"].append(name)
            else:
                # Check if this unknown face matches any previously seen unknown faces
                is_new_unknown = True
                if len(unknown_face_encodings) > 0:
                    unknown_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
                    if True in unknown_matches:
                        is_new_unknown = False
                        # Use existing unknown ID
                        unknown_index = unknown_matches.index(True)
                        name = f"Unknown_{unknown_index + 1}"
                current_detections["unknown"] += 1
                
                if is_new_unknown:
                    # This is a new unknown face
                    unknown_face_counter += 1
                    name = f"Unknown_{unknown_face_counter}"
                    unknown_face_encodings.append(face_encoding)
                    
                    # Only buzz if it's been more than 5 seconds since last alert
                    if current_time - last_unknown_check > 5:
                        threading.Thread(target=buzz, args=(0.5,), daemon=True).start()
                        last_unknown_check = current_time
        
        face_names.append(name)
    
    # Display results with different colors for known/unknown
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Choose color based on whether face is known or unknown
        if name.startswith("Unknown"):
            # Red for unknown faces
            color = (0, 0, 255)  # Red
        else:
            # Green for known faces
            color = (0, 255, 0)  # Green
        
        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Add the current detections summary
    detection_text = []
    if current_detections["known"]:
        detection_text.append(f"Known: {', '.join(current_detections['known'])}")
    if current_detections["unknown"] > 0:
        detection_text.append(f"Unknown: {current_detections['unknown']}")
    
    if detection_text:
        detection_summary = " | ".join(detection_text)
        cv2.putText(frame, detection_summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Direct implementation of capture_photos to avoid import issues
def capture_photos_directly(name, photo_count=10):
    """Direct implementation of photo capture to avoid import issues"""
    try:
        dataset_folder = "dataset"
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        person_folder = os.path.join(dataset_folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        
        # Use the global camera or create a new one
        global camera
        if camera is None:
            camera = init_camera()
            if camera is None:
                return False
        
        print(f"Taking {photo_count} photos for {name}...")
        captured_count = 0
        
        for i in range(photo_count):
            frame = camera.capture_array()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(person_folder, filename)
            cv2.imwrite(filepath, frame)
            captured_count += 1
            print(f"Photo {captured_count}/{photo_count} saved: {filename}")
            time.sleep(0.5)  # Wait between captures
        
        print(f"Photo capture completed. {captured_count} photos saved for {name}.")
        return True
        
    except Exception as e:
        print(f"Error capturing photos: {e}")
        return False


def delete_user_directly(username):
    """Delete a user from the system"""
    try:
        dataset_folder = "dataset"
        user_folder = os.path.join(dataset_folder, username)
        
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            print(f"Deleted user folder: {user_folder}")
            
            # Check if there are any users left
            remaining_users = [name for name in os.listdir(dataset_folder) 
                             if os.path.isdir(os.path.join(dataset_folder, name))]
            
            if len(remaining_users) == 0:
                print("[INFO] No users left - creating empty encodings file")
                # Create empty encodings file
                data = {"encodings": [], "names": []}
                with open("encodings.pickle", "wb") as f:
                    f.write(pickle.dumps(data))
                # Reload empty encodings
                load_encodings()
                return True
            else:
                # Retrain the model with remaining users
                if train_model():
                    return True
                else:
                    return False
        else:
            print(f"User folder not found: {user_folder}")
            return False
            
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False

def delete_all_users_directly():
    """Delete all users and reset the system"""
    try:
        dataset_folder = "dataset"
        
        if os.path.exists(dataset_folder):
            # Remove entire dataset folder
            shutil.rmtree(dataset_folder)
            print("Deleted all user data")
        
        # Create empty encodings file
        data = {"encodings": [], "names": []}
        with open("encodings.pickle", "wb") as f:
            f.write(pickle.dumps(data))
        
        # Reload empty encodings
        load_encodings()
        print("System reset - all users deleted and encodings cleared")
        return True
        
    except Exception as e:
        print(f"Error deleting all users: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global recognition_active
    recognition_active = True
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition')
def recognition():
    global recognition_active, unknown_face_counter, unknown_face_encodings, last_unknown_check
    recognition_active = True
    unknown_face_counter = 0  # Reset counter
    unknown_face_encodings = []  # Clear stored unknown faces
    last_unknown_check = 0  # Reset timer
    load_encodings()  # Reload encodings in case new ones were added
    return render_template('recognition.html')

@app.route('/enroll')
def enroll():
    global recognition_active
    recognition_active = False
    return render_template('enroll.html')

@app.route('/manage')
def manage():
    global recognition_active
    recognition_active = False
    return render_template('manage.html')

@app.route('/capture_photos', methods=['POST'])
def capture_photos():
    try:
        data = request.get_json()
        person_name = data.get('name', 'Unknown')
        
        if not person_name or person_name == 'Unknown':
            return jsonify({'status': 'error', 'message': 'Please enter a valid name'})
        
        # Use the direct implementation to avoid import issues
        success = capture_photos_directly(person_name, photo_count=10)
        
        if success:
            return jsonify({'status': 'success', 'message': f'10 photos captured for {person_name}'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to capture photos'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Run model training in a separate process to prevent segfault from affecting main process
        import subprocess
        import sys
        
        # Create a new Python process to run the training
        result = subprocess.run([sys.executable, 'model_training.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Only reload encodings if training was successful
            if load_encodings():
                return jsonify({'status': 'success', 'message': 'Model trained successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Model trained but failed to load encodings'})
        else:
            error_msg = result.stderr or 'Model training failed'
            return jsonify({'status': 'error', 'message': error_msg})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Training error: {str(e)}'})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        username = data.get('username', '')
        
        if not username:
            return jsonify({'status': 'error', 'message': 'No username provided'})
        
        success = delete_user_directly(username)
        
        if success:
            return jsonify({'status': 'success', 'message': f'User {username} deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to delete user {username}'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    

@app.route('/delete_all_users', methods=['POST'])
def delete_all_users():
    """Delete all users and reset the system"""
    try:
        success = delete_all_users_directly()
        
        if success:
            return jsonify({'status': 'success', 'message': 'All users deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete all users'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})  

@app.route('/current_detections')
def get_current_detections():
    """Get the current face detections"""
    try:
        return jsonify({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'detections': current_detections
        })
    except Exception as e:
        return jsonify({'error': str(e)})  
  

@app.route('/stop_recognition')
def stop_recognition():
    global recognition_active
    recognition_active = False
    return jsonify({'status': 'stopped'})

@app.route('/get_system_status')
def get_system_status():
    """Get current system status including number of registered faces"""
    try:
        # Count actual unique users from the dataset folder
        dataset_path = "dataset"
        unique_users = []
        if os.path.exists(dataset_path):
            unique_users = [name for name in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, name))]
        
        status = {
            'registered_users_count': len(unique_users),
            'registered_users_list': unique_users,
            'face_encodings_count': len(known_face_encodings),
            'recognition_active': recognition_active
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_registered_users')
def get_registered_users():
    try:
        users = []
        dataset_path = "dataset"
        if os.path.exists(dataset_path):
            users = [name for name in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, name))]
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'users': [], 'error': str(e)})
import atexit

@atexit.register
def cleanup_gpio():
    try:
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpiochip_close(h)
        print("[INFO] Cleaned up GPIO safely.")
    except Exception:
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)