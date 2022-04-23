import cv2
import sys

from util import resize_with_aspectratio
from detector import PaintingDetector
from localiser import Localiser

def test():
    if len(sys.argv) != 2:
        raise ValueError('Only provide a path to a video')
    
    video_path = sys.argv[1]
    print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = PaintingDetector()
    localiser = Localiser()

    while True:
        success, img = cap.read()

        if not success:
            break
        
        detector.img = img
        contour_results, img_with_contours = detector.contours(display=False)
        #contour_results_rescaled = detector.scale_contour_to_original_coordinates(contour_results,img_with_contours.shape,img.shape)

        
        room_scores_ordered = localiser.localise(img, contour_results)
        cv2.imshow(room_scores_ordered[0][0], img_with_contours)

        k = cv2.waitKey(int(1000 / fps / 1.5))

        if k != -1:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    test()