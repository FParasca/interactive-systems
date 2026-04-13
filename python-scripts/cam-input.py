import cv2
import cv2.aruco as aruco


def start_cam():
    cap = cv2.VideoCapture(1)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error reading frame")
            break
        cv2.imshow("Camera Input", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    start_cam() 