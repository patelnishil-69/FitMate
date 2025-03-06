from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pushup')
def pushup():
    subprocess.run(["python", "scripts/pushup.py"])
    return render_template('exercise.html', exercise="Pushup")

@app.route('/squat')
def squat():
    subprocess.run(["python", "scripts/squat.py"])
    return render_template('exercise.html', exercise="Squat")

@app.route('/plank')
def plank():
    subprocess.run(["python", "scripts/plank.py"])
    return render_template('exercise.html', exercise="Plank")

@app.route('/shoulder_press')
def shoulder_press():
    subprocess.run(["python", "scripts/shoulder_press.py"])
    return render_template('exercise.html', exercise="Shoulder Press")

if __name__ == '__main__':
    app.run(debug=True)



import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pygame  # For audio playback

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for playing sounds
pygame.mixer.init()

# Dictionary to store last warning time for different errors
last_warning_time = {"posture": 0}
playing_audio = False  # To prevent overlapping sounds

# Function to play warning sound
def play_warning_sound(warning_type):
    global playing_audio
    if playing_audio:
        return  # Avoid playing another sound simultaneously

    playing_audio = True
    sound_path = f"scripts/audiowarn/{warning_type}.mp3"

    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        time.sleep(2)  # Wait for the sound to finish before allowing new ones
    except Exception as e:
        print(f" Error playing sound: {e}")

    playing_audio = False

# Function to calculate angle between three points
def calculate_ang(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0)))

# Initialize Pose model
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)  # Open webcam

# Variables for push-up detection
pushup_counter = 0
stage = "up"
bad_form = False

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

        # Extract necessary body landmarks
        key_points = {
            "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE
        }
        coords = {k: [landmarks[v].x, landmarks[v].y] for k, v in key_points.items()}

        # Calculate angles
        elbow_ang = calculate_ang(coords["shoulder"], coords["elbow"], coords["wrist"])
        body_ang = calculate_ang(coords["shoulder"], coords["hip"], coords["ankle"])

        # Check for improper form
        bad_form = False
        warning_texts = []

        current_time = time.time()  # Get current time

        # Check if back is not straight
        if body_ang < 160 or body_ang > 195:
            bad_form = True
            warning_texts.append("Fix your posture!")
            if current_time - last_warning_time["posture"] >= 2:  # Warn only after 2 sec
                last_warning_time["posture"] = current_time
                threading.Thread(target=play_warning_sound, args=("postwarning",), daemon=True).start()

        # Push-up detection logic
        if elbow_ang < 90 and stage == "up" and not bad_form:
            stage = "down"
        elif elbow_ang > 160 and stage == "down" and not bad_form:
            stage = "up"
            pushup_counter += 1  # Increase push-up count

        # Prepare text data
        text_data = [
            f'Push-Ups: {pushup_counter}',
            f'Elbow Angle: {int(elbow_ang)} deg',
            f'Body Angle: {int(body_ang)} deg'
        ]
        text_data.extend(warning_texts)  # Add warnings if any

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
            color = (0, 0, 255) if "Fix" in text else (0, 0, 0)  # Red for warnings
            cv2.putText(img, text, (20, y_offset + (i * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the img
    img = cv2.resize(img, (1440, 1080))
    cv2.imshow('Push-Up Form Checker', img)

    # Key press options
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r'):  # Reset push-up counter
        pushup_counter = 0

cap.release()
cv2.destroyAllWindows()





import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for sound playback
pygame.mixer.init()

# Function to calculate the angle between three points
def calculate_ang(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    ba = a - b
    bc = c - b
    
    cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_ang, -1.0, 1.0))
    return np.degrees(angle)

# Function to play warning sounds
def play_warning_sound(warning_type):
    """Plays warning sound for back or squat depth issues."""
    sound_path = f"scripts/audiowarn/S_{warning_type}.mp3"
    
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        time.sleep(2)  # Ensure one sound plays at a time
    except Exception as e:
        print(f"⚠️ Error playing sound: {e}")

# Initialize video capture
cap = cv2.VideoCapture(1)

# Variables for squat detection
stage = "up"
sq_ct = 0  # Squat count
bf = False  # Flag for improper form
last_warning_time = {"back": 0, "low": 0}  # Track last warning time

pose = mp_pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for key points
        key_points = {
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER
        }
        coords = {k: [landmarks[v].x, landmarks[v].y] for k, v in key_points.items()}

        # Calculate angles
        hip_ang = calculate_ang(coords["shoulder"], coords["hip"], coords["knee"])
        knee_ang = calculate_ang(coords["hip"], coords["knee"], coords["ankle"])
        back_ang = calculate_ang(coords["shoulder"], coords["hip"], coords["ankle"])

        # Form validation
        bf = False
        warning_texts = []
        current_time = time.time()

        if hip_ang > 100 and knee_ang < 135:  # If squat is too shallow
            bf = True
            warning_texts.append("Go Lower!")
            if current_time - last_warning_time["low"] >= 2:
                last_warning_time["low"] = current_time
                threading.Thread(target=play_warning_sound, args=("low",), daemon=True).start()

        if back_ang < 170 and knee_ang > 160:  # If back is rounded
            bf = True
            warning_texts.append("Keep Back Straight!")
            if current_time - last_warning_time["back"] >= 2:
                last_warning_time["back"] = current_time
                threading.Thread(target=play_warning_sound, args=("back",), daemon=True).start()

        # Squat state detection (Rep counting logic)
        if knee_ang < 100 and stage == "up" and not bf:
            stage = "down"
        elif knee_ang > 160 and stage == "down":
            stage = "up"
            sq_ct += 1  # Count squat

        # Display angles, squat count, and warnings
        text_data = [
            f'Squats: {sq_ct}',
            f'Knee Angle: {int(knee_ang)} deg',
            f'Back Angle: {int(back_ang)} deg',
            f'Hip Angle: {int(hip_ang)} deg',
            f'Stage: {stage}'
        ]
        text_data.extend(warning_texts)

        # Define text box dimensions
        text_size = 30
        padding = 10
        box_width = 400
        box_height = (len(text_data) * text_size) + (2 * padding)

        # Draw white rectangle for text display
        cv2.rectangle(image, (10, 10), (10 + box_width, 10 + box_height), (255, 255, 255), -1)

        # Draw text inside the box
        y_offset = 30  
        for i, text in enumerate(text_data):
            color = (0, 0, 255) if "Go Lower" in text or "Keep Back" in text else (0, 0, 0)  
            cv2.putText(image, text, (20, y_offset + (i * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the image
    image = cv2.resize(image, (1920, 1080))
    cv2.imshow('Squat Detection', image)

    key = cv2.waitKey(10) & 0xFF
    if key== ord('q'):
        break
    elif key == ord('r'):  # Reset rep count
        sq_ct = 0
cap.release()
cv2.destroyAllWindows()
