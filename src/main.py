import cv2
import sys

from util import resize_with_aspectratio
from detector import PaintingDetector
from preprocessing import FrameProcessor
#from localiser import Localiser

def main():
    print(len(sys.argv))
    if len(sys.argv) != 4:
        raise ValueError('Provide paths to the video, calibration file and the database file')
    
    video_path = sys.argv[1]
    calibration_file = sys.argv[2]
    database_file = sys.argv[3]
    print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = PaintingDetector()
    #preproc = FrameProcessor(calibration_file, (width, height))
    #local = Localiser(database_file)

    while True:
        success, img = cap.read()

        if not success:
            break
        
        # For videos taken with GoPro camera
        #img = preproc.undistort(img)

        is_blurred = FrameProcessor.sharpness_metric(img, print_metric=True)
        if is_blurred:
            cv2.imshow("Image", resize_with_aspectratio(img, width=500))
        else:

            detector.img = img
            result, img_with_contours = detector.contours(display=False)

            cv2.imshow("Image", img_with_contours)

            #room_scores_ordered = local.localise(img, result)
            #print(room_scores_ordered)

        k = cv2.waitKey(int(1000 / fps / 1.5))

        if k != -1:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()