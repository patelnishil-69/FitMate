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
        if not pygame.mixer.music.get_busy():  # Ensure no overlap
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"⚠️ Error playing sound: {e}")

# Initialize video capture
cap = cv2.VideoCapture(1)

# Variables for squat detection
stage = "up"
sq_ct = 0  # Squat count
bf = False  # Flag for improper form
warning_start_time = {"back": None, "low": None}  # Track first warning appearance
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

        for issue, condition in [("low", hip_ang > 100 and knee_ang < 135), ("back", back_ang < 170 and knee_ang > 160)]:
            if condition:
                bf = True
                warning_texts.append("Go Lower!" if issue == "low" else "Keep Back Straight!")
                if warning_start_time[issue] is None:
                    warning_start_time[issue] = current_time  # Mark first warning time
                elif current_time - warning_start_time[issue] >= 2:
                    threading.Thread(target=play_warning_sound, args=(issue,), daemon=True).start()
            else:
                warning_start_time[issue] = None  # Reset warning time if corrected

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
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset rep count
        sq_ct = 0

cap.release()
cv2.destroyAllWindows()
