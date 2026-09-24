import cv2

#A capture
cap = cv2.VideoCapture(0)

#loop
while True:
    ret, frame = cap.read()

    if not ret: 
        break

    cv2.imshow("Ashley", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()          # Give camera access back
cv2.destroyAllWindows() # Close OpenCV windows
        