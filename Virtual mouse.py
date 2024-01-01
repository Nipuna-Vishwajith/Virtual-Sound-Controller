import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

# Constants for defining hand landmarks
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP

# Flags to track left-click and drawing gestures
left_click = False
drawing = False
prev_x, prev_y = 0, 0

# Mouse control parameters
mouse_speed = 5
hand_detection_threshold = 0.1  # Adjust this threshold based on your comfort

while True:
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Get coordinates of index finger tip and middle finger tip
            index_finger_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[MIDDLE_FINGER_TIP]

            # Check for left-click gesture (index and middle fingers up)
            left_click = (index_finger_tip.y < middle_finger_tip.y)

        # Perform left-click action
        if left_click:
            pyautogui.mouseDown(button='left')
            drawing = True
        elif drawing:
            pyautogui.mouseUp(button='left')
            drawing = False

            # Reset previous coordinates for smooth drawing
            prev_x, prev_y = 0, 0

        # Draw on canvas if the left-click is pressed
        if drawing:
            x, y = int(index_finger_tip.x * 1920), int(index_finger_tip.y * 1080)

            # Smooth drawing by moving the mouse to the current position
            pyautogui.moveTo(x, y, duration=0.1)

            # Draw a line from the previous position to the current position
            pyautogui.dragTo(x, y, duration=0.1)

            # Update previous coordinates
            prev_x, prev_y = x, y

    # Display the frame with hand landmarks
    cv2.imshow('MediaPipe Hands', image)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
