import cv2
import mediapipe as mp
import time
import csv
import os
from argparse import ArgumentParser

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the Hands model
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def detect_and_draw_hands(frame):
    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    results = hands.process(frame_rgb)
    
    # Initialize an empty list for storing landmarks data
    hand_landmarks_list = []
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the original frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            
            # Extract landmarks
            hand_data = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            hand_landmarks_list.append(hand_data)
    
    return frame, hand_landmarks_list

def main(args):
    
    path_to_data = ".\\data\\"
    idx = args.index
    gesture_name = args.name
    all_hand_landmarks = []
    
    # Start the webcam feed
    cap = cv2.VideoCapture(1)
    
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_frame_time = time.time()
    new_frame_time = 0
    
    save_data = False
    
    try:
        while True:
            # Read frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Perform hand detection and draw landmarks
            frame, hand_landmarks_result = detect_and_draw_hands(frame)
            if len(hand_landmarks_result) > 0:
                hand_landmarks = hand_landmarks_result[0]
            
            if save_data:
                with open(os.path.join(path_to_data, str(idx) + '_' + gesture_name + ".csv"), "a") as out_file:
                    if len(hand_landmarks) == 21:
                        for i in range(20):
                            out_file.write(f"{hand_landmarks[i][0]},{hand_landmarks[i][1]},{hand_landmarks[i][2]},")
                        out_file.write(f"{hand_landmarks[20][0]},{hand_landmarks[20][1]},{hand_landmarks[20][2]}\n")                        
            
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f'FPS: {int(fps)}'

            # Display FPS on the frame
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Hand Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                save_data = not save_data
                if save_data:
                    print("Saving data to disk")
                else:
                    print("Stopped saving data to disk")

            # Press 'q' to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the gesture for which you are capturing photos")
    parser.add_argument("--index", type=int, required=True, help="Index of the gesture for which you are capturing photos")
    args = parser.parse_args()
    main(args)
