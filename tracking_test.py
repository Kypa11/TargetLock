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

    # Run YOLO tracking on people only
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0]
    )

    # Draw the detections and tracking IDs
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("TargetLock - Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()