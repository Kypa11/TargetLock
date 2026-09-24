import cv2

# Asking OpenCV fo the webcam. cap=capture
cap = cv2.VideoCapture(0)

# Adding frames
while True:
    ret, frame = cap.read() #read()=(success, image)

    if not ret:
        break

    cv2.imshow("TargetLock", frame)

    if cv2.waitKey(1) == ord("q"):
        break 


cap.release()          # Give camera access back
cv2.destroyAllWindows() # Close OpenCV windows
