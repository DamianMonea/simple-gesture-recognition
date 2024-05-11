import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HandPoseClassifier(nn.Module):
    def __init__(self):
        super(HandPoseClassifier, self).__init__()
        self.fc1 = nn.Linear(63, 128)  # Input layer to first hidden layer
        self.dropout1 = nn.Dropout(0.25)  # Dropout layer after the first hidden layer
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer
        self.dropout2 = nn.Dropout(0.25)  # Dropout layer after the second hidden layer
        self.fc3 = nn.Linear(128, 64)   # Third hidden layer
        self.dropout3 = nn.Dropout(0.25)  # Dropout layer after the third hidden layer
        self.fc4 = nn.Linear(64, 7)     # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for the first hidden layer
        x = self.dropout1(x)     # Apply dropout
        x = F.relu(self.fc2(x))  # Activation function for the second hidden layer
        x = self.dropout2(x)     # Apply dropout
        x = F.relu(self.fc3(x))  # Activation function for the third hidden layer
        x = self.dropout3(x)     # Apply dropout
        x = self.fc4(x)          # No activation function here for raw scores to go into the loss function
        return x
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

def main():
    
    # Start the webcam feed
    cap = cv2.VideoCapture(1)
    model = HandPoseClassifier()
    model.load_state_dict(torch.load("./model.torch"))
    model.eval()
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_frame_time = time.time()
    new_frame_time = 0
    
    num_to_pose = {
        0: "Open",
        1: "Closed",
        2: "Ok",
        3: "Rock & Roll",
        4: "Thumbs Up",
        5: "Thumbs Down",
        6: "Peace"
    }
    
    try:
        while True:
            # Read frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Perform hand detection and draw landmarks
            frame, hand_landmarks_result = detect_and_draw_hands(frame)
            label = "None"
            if len(hand_landmarks_result) > 0:
                hand_landmarks = hand_landmarks_result[0]
                x = []
                for i in range(21):
                    for j in range(3):
                        x.append(hand_landmarks[i][j])
                output = model(torch.from_numpy(np.array(x).astype(np.float32)))
                prediction = torch.argmax(output).item()
                label = num_to_pose[prediction]
            
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f'FPS: {int(fps)}'

            # Display FPS on the frame
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Current gesture: " + label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

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
    main()
