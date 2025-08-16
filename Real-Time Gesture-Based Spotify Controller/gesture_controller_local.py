import mediapipe as mp
import cv2
import time
import threading
import math
import numpy as np
import pyautogui
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

class Face():
    def __init__(self, nb_faces=1, min_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, 
                                                   max_num_faces=nb_faces, 
                                                   min_detection_confidence=min_conf, 
                                                   min_tracking_confidence=min_conf)
        # Drawing specifications for landmarks and connections
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        self.gesture_controller = GestureController()
    
    def calculate_angle(self, p1, p2):

        v1 = np.array(p2) - np.array(p1)
        v2 = np.array([0, 1])  # Vertical line vector
        angle_rad = np.arctan2(v1[0], v1[1])  # x vs y
        angle_degrees = np.degrees(angle_rad)
        return angle_degrees

    def face_state(self, angle, threshold=20):
        if angle < -threshold:
            return "Left"
        elif angle > threshold:
            return "Right"
        else:
            return "Neutral"

    def detect(self, image_rgb, image_bgr, draw=True):
        face_res = self.face_mesh.process(image_rgb)

        if face_res.multi_face_landmarks:
            for face_landmarks in face_res.multi_face_landmarks:
                h, w, _ = image_bgr.shape

                # Get pixel coordinates of landmarks
                # Forehead (landmarks 10)
                forehead = face_landmarks.landmark[10]
                forehead_coords = (int(forehead.x * w), int(forehead.y * h))

                # Chin (landmark 152)
                chin = face_landmarks.landmark[152]
                chin_coords = (int(chin.x * w), int(chin.y * h))

                #nose tip (landmark 1)
                nose_tip = face_landmarks.landmark[1]
                nose_coords = (int(nose_tip.x * w), int(nose_tip.y * h))

                if draw:
                    # Calculate middle point between forehead and chin for vertical axis line
                    # or draw line from forehead to chin
                    cv2.line(image_bgr, forehead_coords, chin_coords, (255, 255, 0), 2)
                    mp.solutions.drawing_utils.draw_landmarks(
                        image_bgr,
                        face_landmarks,
                        None,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec
                    )
                    #draw infinite line through the nose tip
                    cv2.line(image_bgr, (nose_coords[0], 0), (nose_coords[0], h), (255, 0, 0), 1)

                    # calculate angle between head and vertical line
                    calculated_angle = self.calculate_angle(forehead_coords, nose_coords)
                    cv2.putText(image_bgr, f'Angle: {int(calculated_angle)}', 
                                (forehead_coords[0], forehead_coords[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    state = self.face_state(calculated_angle, threshold=30)
                    print(f"Face State: {state}")
                    self.gesture_controller.update(state)
                    cv2.putText(image_bgr, f'State: {state}', 
                                (forehead_coords[0], forehead_coords[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return image_bgr
    
class Hand():
    def __init__(self, nb_hands=2, min_conf=0.8):
        self.mp_hands = mp.solutions.hands
        # Initialize Hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                       max_num_hands=nb_hands, 
                                       min_detection_confidence=min_conf, 
                                       min_tracking_confidence=min_conf)
        # Drawing specifications for landmarks and connections
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        self.gesture_controller = GestureController()
    
    def right_hand_state(self, hand_landmarks):
        thumb_tip, thumb_base = hand_landmarks.landmark[4].y, hand_landmarks.landmark[2].y
        index_tip,  index_base= hand_landmarks.landmark[8].y, hand_landmarks.landmark[5].y
        middle_tip, middle_base = hand_landmarks.landmark[12].y, hand_landmarks.landmark[9].y
        ring_tip,  ring_base= hand_landmarks.landmark[16].y, hand_landmarks.landmark[13].y
        pinky_tip,  pinky_base= hand_landmarks.landmark[20].y, hand_landmarks.landmark[17].y
        if thumb_tip < thumb_base and index_tip < index_base and middle_tip < middle_base and ring_tip < ring_base and pinky_tip < pinky_base:
            return "High Five"
        elif thumb_tip < index_tip and thumb_tip < middle_tip and thumb_tip < ring_tip and thumb_tip < pinky_tip:
            return "Thumbs Up"
        else:
            return "Other"
    
    def detect(self, image_rgb, image_bgr, draw=True, label_mirror=True):
        hand_res = self.hands.process(image_rgb)

        if hand_res.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                # Swap labels to mirror the handedness
                if label_mirror:
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"
                score = handedness.classification[0].score  # confidence

                h, w, _ = image_bgr.shape
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                
                if draw:
                    if label == "Left":
                        if not self.gesture_controller.vol_adjusting and not self.gesture_controller.vol_blocked:
                            self.gesture_controller.start_vol_adjust_delay()
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        self.gesture_controller.adjust_volume(distance)
                        cv2.putText(image_bgr, f'distance: {distance:.2f}', (int((x1 + x2)/2), int((y1 + y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(image_bgr, f'Left Hand: {score:.2f}', (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.line(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.circle(image_bgr, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
                        cv2.circle(image_bgr, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
                        cv2.circle(image_bgr, (int((x1 + x2)/2), int((y1 + y2)/2)), 5, (255, 0, 0), cv2.FILLED)

                        # Draw landmarks and connections with custom specs
                        mp.solutions.drawing_utils.draw_landmarks(
                            image_bgr,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.drawing_spec,
                            connection_drawing_spec=self.drawing_spec
                        )
                    elif label == "Right":
                        rh_state = self.right_hand_state(hand_landmarks)
                        self.gesture_controller.update(rh_state)
                        cv2.putText(image_bgr, f'Right Hand: {rh_state}', (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(image_bgr, f'Right Hand: {score:.2f}', (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Draw landmarks and connections with custom specs
                        mp.solutions.drawing_utils.draw_landmarks(
                            image_bgr,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.drawing_spec,
                            connection_drawing_spec=self.drawing_spec
                        )
        return image_bgr

class GestureController():
    def __init__(self, min_frames=10, sleep_time=2.0):
        self.min_frames = min_frames
        self.sleep_time = sleep_time
        self.history = []
        self.toggle = True
        self.vol_adjusting = False
        self.vol_blocked = False

    def start_vol_adjust_delay(self):
        if not self.vol_adjusting and not self.vol_blocked:
            threading.Timer(1.0, self.enable_adjustment).start() #prevent immediate adjustment

    def enable_adjustment(self):
        self.vol_adjusting = True
        threading.Timer(3.0, self.block_adjustment).start() 

    def block_adjustment(self):
        self.vol_adjusting = False
        self.vol_blocked = True
        threading.Timer(2.0, self.reset_block).start()

    def reset_block(self):
        self.vol_blocked = False

    def adjust_volume(self, distance, max_distance=250):
        if self.vol_adjusting and not self.vol_blocked:
            self.start_vol_adjust_delay()
            distance = max(0, min(distance, max_distance))
            volume_level = distance / max_distance
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            volume.SetMasterVolumeLevelScalar(volume_level, None)
            print("Adjusting Volume")

    def commands(self, state):
        if state == "High Five":
            pyautogui.press("playpause")
            print("Executing High Five Command")
        elif state == "Thumbs Up":
            pyautogui.press("playpause")
            print("Executing Thumbs Up Command")
        elif state == "Left":
            pyautogui.press("prevtrack")
            print("Executing Left Command")
        elif state == "Right":
            pyautogui.press("nexttrack")
            print("Executing Right Command")

    def lock_toggle(self):
        self.toggle = False
        threading.Timer(self.sleep_time, self.unlock_toggle).start()

    def unlock_toggle(self):
        self.toggle = True
        self.history.clear()
    
    def update(self, state):
        if state in ("Neutral", "Other"):
            self.history.clear()
        elif not self.history or state != self.history[-1]:
            self.history = [state]
        else:
            self.history.append(state)
            if len(self.history) >= self.min_frames and self.toggle:
                self.commands(state)
                self.lock_toggle()
                
                
        


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_time = 0
current_time = 0

# Initialize Face and Hand detectors
face_detector = Face()
hand_detector = Hand()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process both detectors
    image = face_detector.detect(image_rgb, frame)
    image = hand_detector.detect(image_rgb, frame)

    image_rgb.flags.writeable = True
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display FPS on the image
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera Feed', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
