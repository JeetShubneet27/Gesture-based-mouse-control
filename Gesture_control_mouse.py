import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    static_image_mode=False
)
draw_utils = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
frame_scale = 0.75

SMOOTHING_FACTOR = 0.15
CLICK_THRESHOLD = 0.06
DRAG_THRESHOLD = 0.08
SCROLL_ACTIVATION_THRESHOLD = 0.1
DEADZONE = 0.05
DOUBLE_CLICK_TIME = 0.4

class ControlState:
    def __init__(self):
        self.last_click = 0
        self.dragging = False
        self.scrolling = False
        self.click_count = 0
        self.prev_hand_size = 0
        self.scroll_neutral = 0

state = ControlState()

def get_landmark_coords(landmark, frame_w, frame_h):
    return (landmark.x * frame_w, landmark.y * frame_h)

def is_pinch(landmarks, tip_idx, pip_idx, threshold):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    distance = math.hypot(tip.x - pip.x, tip.y - pip.y)
    hand_size = math.hypot(
        landmarks[5].x - landmarks[0].x,
        landmarks[5].y - landmarks[0].y
    )
    return (distance / hand_size) < threshold

def handle_scroll(landmarks, frame_h):
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    middle_pip = landmarks[11]
    ring_pip = landmarks[15]

    middle_bent = middle_tip.y > middle_pip.y
    ring_bent = ring_tip.y > ring_pip.y

    if middle_bent and ring_bent:
        current_pos = (middle_tip.y + ring_tip.y) / 2
        wrist_pos = landmarks[0].y
        
        if not state.scrolling:
            state.scroll_neutral = wrist_pos
            state.scrolling = True
        
        scroll_amount = (current_pos - state.scroll_neutral) * 2.5
        pyautogui.scroll(int(-scroll_amount * 50))
        
        state.scroll_neutral += scroll_amount * 0.1
    else:
        state.scrolling = False

def handle_controls(landmarks, frame_w, frame_h):
    wrist = landmarks[0]
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]

    hand_vector = (index_tip.x - wrist.x, index_tip.y - wrist.y)
    delta_x = hand_vector[0] * (1 + 2 * abs(hand_vector[0]))
    delta_y = hand_vector[1] * (1 + 2 * abs(hand_vector[1]))
    
    screen_x = int((0.5 + delta_x * (1 - DEADZONE*2)) * screen_w)
    screen_y = int((0.5 + delta_y * (1 - DEADZONE*2)) * screen_h)
    
    current_x, current_y = pyautogui.position()
    new_x = SMOOTHING_FACTOR * screen_x + (1 - SMOOTHING_FACTOR) * current_x
    new_y = SMOOTHING_FACTOR * screen_y + (1 - SMOOTHING_FACTOR) * current_y
    pyautogui.moveTo(new_x, new_y)

    if is_pinch(landmarks, 8, 4, CLICK_THRESHOLD):
        current_time = time.time()
        time_since_last = current_time - state.last_click
        
        if time_since_last < DOUBLE_CLICK_TIME:
            pyautogui.doubleClick()
            state.click_count = 0
        else:
            pyautogui.click()
            state.click_count += 1
        
        state.last_click = current_time
        time.sleep(0.1)

    elif is_pinch(landmarks, 8, 4, DRAG_THRESHOLD):
        if not state.dragging:
            pyautogui.mouseDown()
            state.dragging = True
    else:
        if state.dragging:
            pyautogui.mouseUp()
            state.dragging = False

    if is_pinch(landmarks, 20, 4, CLICK_THRESHOLD):
        pyautogui.rightClick()
        time.sleep(0.2)

    handle_scroll(landmarks, frame_h)

cap = cv2.VideoCapture(0)
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_frame_time < 1/30:
        continue
    last_frame_time = time.time()

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hand_detector.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = hand.landmark
        
        draw_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        handle_controls(landmarks, frame_w, frame_h)

    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
