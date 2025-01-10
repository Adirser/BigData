# from ultralytics import YOLO
# import cv2

# # Load YOLOv8 model (pre-trained for general object detection)
# model = YOLO('yolov8n.pt')  # Use yolov8m.pt or yolov8x.pt for higher accuracy

# # Load the video
# video_path = 'friends_video.mp4'  # Replace with your video path
# video = cv2.VideoCapture(video_path)
# output = cv2.VideoWriter('people_tracking.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, 
#                          (int(video.get(3)), int(video.get(4))))

# # Loop through video frames
# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform inference with YOLOv8
#     results = model.predict(source=frame, save=False, conf=0.5)  # Adjust `conf` for confidence threshold
#     detections = results[0].boxes  # Get detected bounding boxes

#     # Draw bounding boxes and confidence scores
#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#         confidence = box.conf[0]  # Confidence score

#         # Draw rectangle and label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Write the frame to output video
#     output.write(frame)

#     # Display the frame (optional)
#     cv2.imshow('YOLOv8 Face Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# video.release()
# output.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition
import numpy as np

# Load the video
video = cv2.VideoCapture('friends_video.mp4')  # Replace with your video path
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, 
                         (int(video.get(3)), int(video.get(4))))

# Process every nth frame to improve performance
FRAME_INTERVAL = 5  # Adjust this to skip frames for faster processing
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_INTERVAL != 0:
        # Skip this frame to save computation
        continue

    # Convert frame to RGB (face_recognition expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')  # CNN-based detection
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Draw rectangles and display probabilities for each detected face
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Calculate probabilities based on face distances (inverse relationship)
        distances = face_recognition.face_distance(face_encodings, face_encodings[i])  # Comparing face to itself
        probability = 1 - np.mean(distances)  # Approximation of confidence

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), colors[i % len(colors)], 2)

        # Put probability near the rectangle
        label = f"Confidence: {probability * 100:.2f}%"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)

    # Write the frame to output video
    output.write(frame)

    # Display the frame (optional)
    cv2.imshow('Face Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
output.release()
cv2.destroyAllWindows()
