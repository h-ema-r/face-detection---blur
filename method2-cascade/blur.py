import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to your video file
video_path = 'people_video_1.mp4'

# Output video file path
output_path = 'all-blurred_output-cascade.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 25, (1020, 600))

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

     # Resize frame
    frame = cv2.resize(frame, (1020, 600))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 35))

    # Blur faces
    for (x, y, w, h) in faces:
        # Apply blur effect to the detected face region
        blurred_face = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face
    
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