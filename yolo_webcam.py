import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO on the current frame
    results = model(frame)

    # Draw the detections
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("TargetLock - YOLO", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()