import cv2
import mediapipe as mp
import numpy as np

def get_eye_aspect_ratio(eye_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in face_landmarks.landmark]

                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                left_ear = get_eye_aspect_ratio(LEFT_EYE, landmarks)
                right_ear = get_eye_aspect_ratio(RIGHT_EYE, landmarks)
                ear = (left_ear + right_ear) / 2.0

                # Gaze direction check
                nose_x = landmarks[1][0]
                mid_eye_x = (landmarks[33][0] + landmarks[362][0]) / 2
                gaze_diff = abs(nose_x - mid_eye_x)

                # Balanced thresholds
                EAR_THRESHOLD = 0.20
                GAZE_TOLERANCE = 50

                if ear > EAR_THRESHOLD and gaze_diff < GAZE_TOLERANCE:
                    status = "Focused"
                    color = (0, 255, 0)
                else:
                    status = "Not Focused"
                    color = (0, 0, 255)

                for idx in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, landmarks[idx], 2, color, -1)

                x, y = landmarks[10]
                cv2.putText(frame, status, (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.imshow("Focus Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
