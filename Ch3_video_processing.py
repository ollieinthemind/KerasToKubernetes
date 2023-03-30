import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture('data\sample_video.mp4')

counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    counter += 1
    print(counter)
    new_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    if counter%30 == 0:
        plt.imshow(new_frame)
        plt.show()

cap.release()

