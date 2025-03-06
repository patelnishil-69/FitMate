import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pygame  # New sound library

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for playing sounds
pygame.mixer.init()

# Audio management variables
last_warning_time = {"hips": 0, "body": 0}  # Tracks when warnings were last given
playing_audio = False  # Prevents multiple audio overlaps

# Function to play warning sound (ensures only one plays at a time)
def play_warning_sound(warning_type):
    global playing_audio
    
    if playing_audio:
        return  # Don't play if another sound is playing

    playing_audio = True
    sound_path = f"scripts/audiowarn/P_{warning_type}.mp3"

    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        time.sleep(2)  # Wait for the sound to finish before allowing new ones
    except Exception as e:
        print(f"⚠️ Error playing sound: {e}")

    playing_audio = False

# Function to calculate angle between three points
def calculate_ang(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0)))

# Initialize Pose estimation
pose = mp_pose.Pose()
cap = cv2.VideoCapture(1)  # Open webcam

plank_timer = 0  # Counter for plank hold time
state = False  # Plank position state
bad_form = False  # Flag for improper form

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key body landmarks for plank tracking
        key_points = {
            "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "wrist": mp_pose.PoseLandmark.LEFT_WRIST
        }
        coords = {k: [landmarks[v].x, landmarks[v].y] for k, v in key_points.items()}

        # Calculate angles
        torso_ang = calculate_ang(coords["shoulder"], coords["hip"], coords["ankle"])  # Ensures straight body
        hip_ang = calculate_ang(coords["shoulder"], coords["hip"], coords["knee"])  # Detects hip sagging

        # Check for improper form
        bad_form = False
        warning_texts = []  

        current_time = time.time()  # Get current time

        if torso_ang < 165 or torso_ang > 195:  # If back isn't straight
            bad_form = True
            warning_texts.append("Keep your body straight!")
            if current_time - last_warning_time["body"] >= 2:  # Only play sound after 2 seconds
                last_warning_time["body"] = current_time
                threading.Thread(target=play_warning_sound, args=("body",), daemon=True).start()

        if hip_ang < 160 or hip_ang > 200:  # Hips too high or low
            bad_form = True
            warning_texts.append("Adjust your hips!")
            if current_time - last_warning_time["hips"] >= 2:  
                last_warning_time["hips"] = current_time
                threading.Thread(target=play_warning_sound, args=("hips",), daemon=True).start()

        # Plank detection logic
        if not bad_form:
            if not state:
                state = True  
            plank_timer += 1  
        else:
            state = False  

        # Prepare text data
        text_data = [
            f'Plank Time: {plank_timer // 30} sec',
            f'Torso Angle: {int(torso_ang)} deg',
            f'Hip Angle: {int(hip_ang)} deg'
        ]
        text_data.extend(warning_texts)  

        # Define text box dimensions
        text_size = 30
        padding = 10
        box_width = 350
        box_height = (len(text_data) * text_size) + (2 * padding)

        # Draw white rectangle for text display
        cv2.rectangle(img, (10, 10), (10 + box_width, 10 + box_height), (255, 255, 255), -1)

        # Draw text inside the box
        y_offset = 30  
        for i, text in enumerate(text_data):
            color = (0, 0, 255) if "Adjust" in text or "Keep" in text else (0, 0, 0)  # Red for warnings
            cv2.putText(img, text, (20, y_offset + (i * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the img
    img = cv2.resize(img, (1920, 1080))
    cv2.imshow('Plank Detection', img)

    # Key press options
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r'):  # Reset plank timer
        plank_timer = 0

cap.release()
cv2.destroyAllWindows()
