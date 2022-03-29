import cv2
import sys

from util import resize_with_aspectratio

def main():
    if len(sys.argv) != 2:
        raise ValueError('Only provide a path to a video')
    
    video_path = sys.argv[1]
    print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    pTime = 0

    while True:
        success, img = cap.read()

        if not success:
            break

        cv2.imshow("Image", resize_with_aspectratio(image=img, width=400))

        cv2.waitKey(1)

if __name__ == '__main__':
    main()