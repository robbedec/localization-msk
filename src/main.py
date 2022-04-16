import cv2
import sys

from util import resize_with_aspectratio
from detector import PaintingDetector

def main():
    if len(sys.argv) != 2:
        raise ValueError('Only provide a path to a video')
    
    video_path = sys.argv[1]
    print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = PaintingDetector()

    while True:
        success, img = cap.read()

        if not success:
            break
        
        detector.img = img
        result, img_with_contours = detector.contours(display=False)

        cv2.imshow("Image", img_with_contours)

        cv2.waitKey(int(1000 / fps / 1.5))

if __name__ == '__main__':
    main()