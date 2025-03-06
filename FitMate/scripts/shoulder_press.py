import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time

# Initialize Pygame Mixer
pygame.mixer.init()

# Function to play warning sound
def play_warning_sound(warning_type):
    """Plays warning sound for arm position issues."""
    sound_path = f"scripts/audiowarn/{warning_type}.mp3"
    
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        time.sleep(2)  # Prevent overlapping sounds
    except Exception as e:
        print(f"Error playing sound: {e}")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(1)  # Open webcam
rep_count, stage = 0, "down"

# Track last warning times to avoid frequent sound playback
last_warning_time = {"level_arms": 0, "bring_arms_closer": 0}

# Function to calculate the angle between three points
def calculate_ang(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    warning_texts = []
    current_time = time.time()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key landmarks for shoulder press
        key_points = {
            "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
            "nose": mp_pose.PoseLandmark.NOSE
        }

        coords = {k: [landmarks[v].x, landmarks[v].y] for k, v in key_points.items()}

        left_elbow_ang = calculate_ang(coords["left_shoulder"], coords["left_elbow"], coords["left_wrist"])
        right_elbow_ang = calculate_ang(coords["right_shoulder"], coords["right_elbow"], coords["right_wrist"])

        left_wrist_above_head = coords["left_wrist"][1] < coords["nose"][1]
        right_wrist_above_head = coords["right_wrist"][1] < coords["nose"][1]

        # Form correction messages and audio triggers
        if abs(left_elbow_ang - right_elbow_ang) > 20:
            warning_texts.append("Level both arms!")
            if current_time - last_warning_time["level_arms"] >= 3:
                last_warning_time["level_arms"] = current_time
                threading.Thread(target=play_warning_sound, args=("SP_lvl",), daemon=True).start()

        if not left_wrist_above_head and not right_wrist_above_head and left_elbow_ang > 50 and right_elbow_ang > 50:
            warning_texts.append("Bring arms closer!")
            if current_time - last_warning_time["bring_arms_closer"] >= 3:
                last_warning_time["bring_arms_closer"] = current_time
                threading.Thread(target=play_warning_sound, args=("SP_close",), daemon=True).start()

        # Shoulder press movement detection
        if left_wrist_above_head and right_wrist_above_head and left_elbow_ang > 160 and right_elbow_ang > 160:
            stage = "up"
        if left_elbow_ang < 90 and right_elbow_ang < 90 and stage == "up":  
            stage, rep_count = "down", rep_count + 1  

        # Draw UI box
        box_height = 140 + (len(warning_texts) * 30)
        cv2.rectangle(image, (10, 10), (350, box_height), (255, 255, 255), -1)

        # Display rep count & angles
        text_data = [
            f'Reps: {rep_count}',
            f'Left Elbow Angle: {int(left_elbow_ang)} deg',
            f'Right Elbow Angle: {int(right_elbow_ang)} deg',
            f'STAGE: {stage}'
        ]

        # Draw text data
        y_offset = 35
        for text in text_data:
            cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 30

        # Draw warning messages
        for warning in warning_texts:
            cv2.putText(image, warning, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Shoulder Press Detection', cv2.resize(image, (1920, 1080)))

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r'):  # Reset rep count
        rep_count = 0

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
