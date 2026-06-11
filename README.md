# Face Recognition Web Application

A Flask-based web application for real-time facial recognition with enrollment capabilities, built for Raspberry Pi with camera support.

## Features

### Core Functionality
- **Real-Time Face Recognition**: Live video streaming with face detection and identification
- **User Enrollment**: Capture and register new faces for the system
- **Known Face Detection**: Identify registered users with green bounding boxes
- **Unknown Face Detection**: Alert system for unregistered faces with red bounding boxes
- **Alarm System**: Audio buzzer alert triggers on unknown face detection
- **User Management**: Add, delete, or reset registered users

### Technical Highlights
- Multi-threaded frame processing for smooth video streaming (~30 FPS)
- Face encoding storage using pickle for fast comparison
- GPIO integration for buzzer alerts
- Two-mode video streaming (recognition vs. preview)
- Enrollment mode with alarm suppression during user registration
- Face recognition model with configurable tolerance (0.5 default for high accuracy)

## System Architecture

### Core Components

#### `app.py` (Main Application)
Flask web server that handles:
- Web interface routing
- Real-time video feed streaming via `/video_feed`
- Face recognition processing with frame-by-frame analysis
- User enrollment and photo capture workflow
- Model training orchestration
- GPIO control for buzzer alerts
- System status management

#### `model_training.py` (Model Training)
Handles face encoding generation:
- Processes all images in the `dataset/` directory
- Generates face encodings for each registered user
- Saves encoded data to `encodings.pickle`
- Supports both HOG and CNN face detection models
- Filters invalid or problematic images

#### `facial_recognition.py` (Reference Implementation)
Standalone face recognition script demonstrating:
- Camera initialization and frame capture
- Face location detection and encoding comparison
- Real-time FPS calculation
- Can be run independently for testing

#### `image_capture.py` (Image Capture Utility)
Helper module for photo capture functionality during enrollment.

### Web Interface
- **Index** (`/`): Homepage and navigation
- **Recognition** (`/recognition`): Live face recognition view
- **Enrollment** (`/enroll`): Add new users to the system
- **Management** (`/manage`): Manage registered users

## Requirements

### Hardware
- Raspberry Pi (recommended Pi 4 or higher)
- PiCamera2 compatible camera module
- GPIO-compatible buzzer (Pin 17 by default)

### Python Dependencies
```
Flask
OpenCV (cv2)
face_recognition
face-recognition-models
numpy
Pillow (PIL)
imutils
picamera2
lgpio
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Dapacifism/10_face_rec_web.git
cd 10_face_rec_web
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or install manually:
pip install Flask opencv-python face_recognition numpy imutils
```

### 3. Configure GPIO (if using buzzer)
Ensure lgpio is installed for GPIO control:
```bash
sudo apt-get install libgpiod2
pip install lgpio
```

### 4. Directory Structure
The application will create the following structure:
```
10_face_rec_web/
├── app.py
├── model_training.py
├── facial_recognition.py
├── image_capture.py
├── templates/
│   ├── index.html
│   ├── recognition.html
│   ├── enroll.html
│   └── manage.html
├── static/
│   └── css/
├── dataset/
│   ├── user1/
│   ├── user2/
│   └── ...
├── encodings.pickle
└── .gitignore
```

## Usage

### Starting the Application
```bash
python app.py
```

The web server will start on `http://localhost:5000` (or your Raspberry Pi's IP address:5000)

### Workflow

#### 1. Enroll a New User
1. Navigate to `/enroll`
2. Enter the user's name
3. Click "Start Enrollment" to begin photo capture
4. 10 photos will be automatically captured (adjust `photo_count` in code as needed)
5. Photos are saved to `dataset/<username>/`
6. Click "Train Model" to generate face encodings
7. System will process all images and save encodings

#### 2. Monitor Recognition
1. Navigate to `/recognition`
2. Live video feed displays with:
   - **Green boxes**: Known faces with names
   - **Red boxes**: Unknown faces with alert
   - **Detection summary**: Shows current detections per frame
3. Buzzer sounds on unknown face detection (3-second alarm with 3-second cooldown)

#### 3. Manage Users
1. Navigate to `/manage`
2. View all registered users
3. Delete individual users or reset the entire system
4. Deleting users triggers automatic model retraining

## API Endpoints

### Video Streaming
- `GET /video_feed?mode=recognition` - Stream video with recognition
- `GET /video_feed?mode=preview` - Stream raw video without processing

### Recognition Control
- `GET /recognition` - Start recognition mode
- `GET /stop_recognition` - Stop recognition and disable frame processing
- `GET /current_detections` - Get real-time detection data

### Enrollment
- `POST /start_enrollment` - Begin enrollment (suppresses alarms)
- `POST /stop_enrollment` - End enrollment mode
- `GET /enroll` - Enrollment interface
- `POST /capture_photos` - Capture photos for a user (JSON: `{"name": "username"}`)

### Model Management
- `POST /train_model` - Train model on dataset images
- `GET /get_system_status` - Get registered user count and status

### User Management
- `GET /manage` - User management interface
- `POST /delete_user` - Delete specific user (JSON: `{"username": "name"}`)
- `POST /delete_all_users` - Delete all users and reset system
- `GET /get_registered_users` - Get list of registered users

## Configuration

### Key Settings (in `app.py`)

```python
BUZZER_PIN = 17                    # GPIO pin for buzzer (BCM numbering)
photo_count = 10                   # Photos captured per user during enrollment
tolerance = 0.5                    # Face matching tolerance (lower = stricter)
alarm_duration = 3                 # Alarm duration in seconds
cooldown = 3                       # Cooldown between alarms
```

### Camera Settings (in `app.py`)

```python
main={"size": (1640, 1232), "format": "RGB888"}  # Resolution and format
```

## Recognition Accuracy

### Factors Affecting Accuracy
- **Photo Quality**: Clear, well-lit photos during enrollment improve recognition
- **Tolerance Setting**: Default 0.5 is strict; increase to 0.6 for more lenient matching
- **Number of Photos**: More enrollment photos = better model accuracy
- **Lighting Conditions**: Consistent lighting between enrollment and recognition

### Tuning Recognition
- Increase tolerance in `process_frame_for_recognition()` for more matches (line 154)
- Capture more photos per user for better coverage
- Use the "large" model in `model_training.py` for more accurate encodings

## Troubleshooting

### Camera Issues
- Ensure camera is enabled: `sudo raspi-config` → Interfacing Options → Camera
- Verify PiCamera2 installation and compatibility
- Check camera is physically connected

### Recognition Not Working
- Verify `encodings.pickle` exists and is populated
- Check that users are properly enrolled in `dataset/` directory
- Ensure lighting conditions are adequate
- Increase tolerance threshold if faces aren't being recognized

### GPIO/Buzzer Issues
- Verify buzzer is connected to GPIO pin 17 (or update `BUZZER_PIN`)
- Check lgpio is properly installed
- Test GPIO with: `lgpio --help`

### Performance Issues
- Lower camera resolution in configuration
- Reduce FPS target (increase `time.sleep()` in `generate_frames()`)
- Use HOG model instead of CNN in `model_training.py` for speed

## Project Statistics

- **Python**: 54.9% (Core logic and backend)
- **HTML**: 32.4% (Web interface)
- **CSS**: 12.7% (Styling)

## License

No license specified. Created by Dapacifism.

## Notes

- The application is designed to run on Raspberry Pi with PiCamera2
- GPIO operations require root privileges for production use
- The system automatically creates necessary directories during first run
- All face encodings are stored locally in `encodings.pickle`
- The application supports multiple concurrent enrollments

## Future Enhancements

Potential improvements:
- Database integration for user management
- Web-based photo upload for enrollment
- Advanced analytics and logging
- Face database export/import
- Multi-camera support
- REST API for third-party integration
- Authentication for web interface
