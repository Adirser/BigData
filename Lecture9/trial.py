import cv2

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the video
video = cv2.VideoCapture('friends_video.mp4')  # Replace with your video path
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, 
                         (int(video.get(3)), int(video.get(4))))

# Loop through video frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to grayscale (required by Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles with unique colors for each face
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], 2)

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
