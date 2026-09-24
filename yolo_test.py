import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Capture one frame
ret, frame = cap.read()

# Give the frame to YOLO
results = model(frame)

# Get the first result
result = results[0]

# Draw YOLO's detections on the frame
annotated_frame = result.plot()

# Display the frame
cv2.imshow("YOLO Detection", annotated_frame)

# Wait for a key press
cv2.waitKey(0)

# Clean up
cap.release()
cv2.destroyAllWindows()