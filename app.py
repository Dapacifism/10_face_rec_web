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
enrollment_in_progress = False  # When True, suppress alarm and unknown-encoding updates

def buzz(duration=1):
    """Activate buzzer for a short duration"""
    try:
        lgpio.gpio_write(h, BUZZER_PIN, 1)
        time.sleep(duration)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
    except Exception as e:
        print(f"[BUZZER ERROR] {e}")


def alarm_buzz(duration=3, on_interval=0.5, off_interval=0.2):
    """Run an alarm-style buzzer: cycles on/off for the given duration.
    This runs in its own thread so it won't block frame processing.
    """
    try:
        start = time.time()
        while time.time() - start < duration:
            lgpio.gpio_write(h, BUZZER_PIN, 1)
            time.sleep(on_interval)
            lgpio.gpio_write(h, BUZZER_PIN, 0)
            # If remaining time is small, break to avoid extra sleep
            if time.time() - start + off_interval >= duration:
                break
            time.sleep(off_interval)
    except Exception as e:
        print(f"[ALARM BUZZ ERROR] {e}")


# Initialize camera
def init_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"},
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
    
    # Always attempt to stream frames. Recognition is applied only when
    # recognition_active is True. Using a perpetual loop lets preview mode
    # stream frames even when recognition is disabled.
    while True:
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
        # Default to unknown
        name = "Unknown"

        # If we have known encodings, try to match. If there are none, we'll
        # treat every face as unknown (useful after deleting all users).
        matched_known = False
        if len(known_face_encodings) > 0:
            # Use stricter tolerance (0.5 or lower is more accurate, 0.6 is default but too loose)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                # Only accept if it matches AND the distance is below threshold (0.5 is recommended)
                if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                    name = known_face_names[best_match_index]
                    matched_known = True
                    if name not in current_detections["known"]:
                        current_detections["known"].append(name)

        # Handle unknown faces (either no known encodings or no match)
        if not matched_known:
                # Check if this unknown face matches any previously seen unknown faces
                is_new_unknown = True
                if not enrollment_in_progress:
                    if len(unknown_face_encodings) > 0:
                        unknown_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
                        if True in unknown_matches:
                            is_new_unknown = False
                            unknown_index = unknown_matches.index(True)
                            name = f"Unknown_{unknown_index + 1}"

                current_detections["unknown"] += 1

                if is_new_unknown and not enrollment_in_progress:
                    unknown_face_counter += 1
                    name = f"Unknown_{unknown_face_counter}"
                    unknown_face_encodings.append(face_encoding)

                # Trigger alarm buzz for unknown face detections unless we're enrolling.
                if not enrollment_in_progress:
                    # Use a cooldown so we don't continuously spawn alarm threads every frame.
                    alarm_duration = 3
                    cooldown = alarm_duration
                    if current_time - last_unknown_check > cooldown:
                        threading.Thread(target=alarm_buzz, args=(alarm_duration, 0.4, 0.15), daemon=True).start()
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
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
    
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
def capture_photos_directly(name, photo_count=20):
    """Direct implementation of photo capture to avoid import issues"""
    try:
        global enrollment_in_progress
        enrollment_in_progress = True
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
    finally:
        try:
            enrollment_in_progress = False
        except Exception:
            pass


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
    # Allow clients to request the feed in different modes. When mode is
    # 'recognition' the frame processing will run recognition; when mode is
    # anything else (for example 'preview' while enrolling) recognition is
    # disabled and frames are streamed raw.
    mode = request.args.get('mode', 'recognition')
    recognition_active = True if mode == 'recognition' else False
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_enrollment', methods=['POST', 'GET'])
def start_enrollment():
    """Mark enrollment as in-progress so alarms and unknown-encoding updates
    are suppressed while the user is enrolling."""
    global enrollment_in_progress, recognition_active, unknown_face_encodings, unknown_face_counter, last_unknown_check
    enrollment_in_progress = True
    # Ensure recognition processing is disabled while enrolling
    recognition_active = False
    # Clear transient unknown state so enrollment isn't affected
    unknown_face_encodings = []
    unknown_face_counter = 0
    last_unknown_check = 0
    return jsonify({'status': 'ok', 'enrollment': 'started'})


@app.route('/stop_enrollment', methods=['POST', 'GET'])
def stop_enrollment():
    """Clear the enrollment flag so normal recognition and alarms resume."""
    global enrollment_in_progress
    enrollment_in_progress = False
    return jsonify({'status': 'ok', 'enrollment': 'stopped'})

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