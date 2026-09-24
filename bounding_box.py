import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.rectangle(
        frame,
        (100, 100),
        (300, 400),
        (255, 255, 0),
        2
    )

    cv2.imshow("TargetLock", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()