import cv2
from ultralytics import YOLO

# Load YOLO model for face detection
facemodel = YOLO('yolov8n-face.pt')

# Path to video file
video_path = 'people_video_1.mp4'

# Output video file path
output_path = 'single-face-blur-yolov.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (1020, 600))

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables for storing coordinates of the tracked faces
tracked_faces = []

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

    # If faces have been tracked, blur the fourth detected face
    if len(tracked_faces) >= 4:
        x, y, w, h = tracked_faces[3]  # Selecting the fourth detected face
        blurred_face = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # If no faces have been tracked yet, search for faces and track the fourth detected face
    elif len(tracked_faces) < 4:
        for info in face_results:
            parameters = info.boxes
            if len(parameters) > 3:  # Check if there are at least four faces detected
                for i in range(3, len(parameters)):  # Start from the fourth face
                    box = parameters[i].xyxy[0]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    tracked_faces.append((x1, y1, x2 - x1, y2 - y1))
                    # Blur the fourth detected face
                    blurred_face = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face
                    break

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with the tracked face blurred
    cv2.imshow('Blurred Faces', frame)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, release the VideoWriter, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()

print("Output video saved successfully.")
