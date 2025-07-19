from flask import Flask, render_template, Response
import cv2
import pyttsx3
from deepface import DeepFace
import mediapipe as mp
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_output(text):
    engine.say(text)
    engine.runAndWait()

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except:
        return "No face"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def detect_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    gesture = "No Hand"
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = handLms.landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))
            if distance < 0.05:
                gesture = "OK"
            else:
                gesture = "Hand Detected"
    return gesture

def detect_lip_movement(prev_lips, current_lips):
    if not prev_lips or not current_lips:
        return "Unknown"
    diff = np.linalg.norm(np.array(prev_lips) - np.array(current_lips))
    return "Speaking" if diff > 0.01 else "Not Speaking"

prev_lips = None

def generate_frames():
    global prev_lips
    while True:
        success, frame = camera.read()
        if not success:
            break

        gesture = detect_gesture(frame)
        emotion = detect_emotion(frame)

        height, width, _ = frame.shape
        lips = []

        mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face.process(img_rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                lips = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in range(61, 69)]
                mp_draw.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_LIPS)

        lip_status = detect_lip_movement(prev_lips, lips)
        prev_lips = lips

        text = f"Expression: {emotion} | Gesture: {gesture} | Lips: {lip_status}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if emotion != "No face":
            speak_output(f"You seem {emotion}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    speak_output("Hello Ashab sir! Your voice system is working perfectly.")
    app.run(debug=True)
