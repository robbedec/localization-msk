import cv2
import sys
from matcher import PaintingMatcher

from util import resize_with_aspectratio
from detector import PaintingDetector
from localiser import Localiser

def test(video_path,database,csv_path):
    # if len(sys.argv) != 2:
    #     raise ValueError('Only provide a path to a video')
    
    # video_path = sys.argv[1]
    # print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = PaintingDetector()
    matcher = PaintingMatcher(csv_path,database)
    localiser = Localiser(matcher=matcher)

    while True:
        success, img = cap.read()

        if not success:
            break
        
        detector.img = img
        contour_results, img_with_contours = detector.contours(display=False)
        #contour_results_rescaled = detector.scale_contour_to_original_coordinates(contour_results,img_with_contours.shape,img.shape)

        room_prediction = localiser.localise_use_all(img, contour_results)
        txt = "Zaal: " + room_prediction
        cv2.putText(img=img_with_contours, text=txt, org=(50, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
        cv2.imshow("video", img_with_contours)

        k = cv2.waitKey(int(1000 / fps / 1.5))

        if k != -1:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('Only provide a path to a video')
    
    
    video_path = sys.argv[1]
    database = sys.argv[2]
    csv_path = sys.argv[3]



    test(video_path,database,csv_path)