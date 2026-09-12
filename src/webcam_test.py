import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("Thai Sign Assist - Webcam Test", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
