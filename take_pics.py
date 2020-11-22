import cv2
import time

cam = cv2.VideoCapture('rtsp://link')
photo_idx = 0
last_time = time.time()
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    if time.time() - last_time > 2:
        print(f"Photo {photo_idx + 1} taken!")
        last_time = time.time()
        cv2.imwrite('./chess_markers/' + str(photo_idx) + '.jpg', frame)
        photo_idx += 1

    cv2.imshow("Result", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

