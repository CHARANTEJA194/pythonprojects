import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip for mirror view
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            index_finger = landmarks[8]  # Tip of the index finger
            x = int(index_finger.x * frame_w)
            y = int(index_finger.y * frame_h)

            # Draw a circle on the tip
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            # Move mouse
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Optional: pinch gesture to click
            thumb_tip = landmarks[4]
            thumb_x = int(thumb_tip.x * frame_w)
            thumb_y = int(thumb_tip.y * frame_h)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 255), -1)

            # Check distance between thumb and index finger
            if abs(x - thumb_x) < 300 and abs(y - thumb_y) < 30:
                pyautogui.click()
                pyautogui.sleep(0.3)  # Delay to avoid multiple clicks

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
