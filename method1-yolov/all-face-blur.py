import cv2
from ultralytics import YOLO

# Load YOLO model for face detection
facemodel = YOLO('yolov8n-face.pt')

# Path to your video file
video_path = 'people_video_1.mp4'

# Output video file path
output_path = 'all-blurred_output-yolov8.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 25, (1020, 600))

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 600))

    # Detect faces using YOLO
    face_results = facemodel.predict(frame, conf=0.40)

    # Blur faces
    for info in face_results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Apply blur effect to the detected face region
            blurred_face = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred_face

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with blurred faces
    cv2.imshow('Blurred Faces', frame)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
print("Output video saved successfully.")
