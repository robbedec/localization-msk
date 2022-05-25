from ast import Mod
import cv2
import sys
import numpy as np
import pandas as pd

from util import resize_with_aspectratio, vertices, room_center_coords
from detector import PaintingDetector
from matcher import PaintingMatcher
from localiser import Localiser
from preprocessing import FrameProcessor
from enum import Enum


class Mode(Enum):
    ORB = 0
    FVECTOR = 1
    COMBINATION = 2

DIFF_ROOM_COUNTER = 0
CURRENT_ROOM = 'Z'

def create_map(room_pred, plan, file_path, visited_rooms):
    # TODO: remove after room_pred are normalized.
    # Random percentages
    #room_pred = np.random.uniform(low=0, high=1, size=(len(vertices),))

    # Load contour data from storage
    df_poly = pd.DataFrame(data=np.load(file_path, allow_pickle=True), columns=['polygon'])
    mask = np.zeros(plan.shape, dtype=np.uint8)


    for i, row in df_poly.iterrows():
        points = np.array(row['polygon'])

        # For color
        pct = room_pred[i]
        pct_diff = 1.0 - pct
        red_color = min(255, pct_diff*2 * 255)
        green_color = min(255, pct*2 * 255)
        col = (0, green_color, red_color)

        poly_filled = cv2.fillPoly(mask, [points], col)
        mask = poly_filled

    blended_im = cv2.addWeighted(plan/255, 0.5, poly_filled/255, 0.5, 0)

    probs_indices_sorted = np.flip(np.argsort(np.array(room_pred)))
    pred_text = [f'Zaal: {vertices[probs_indices_sorted[i]]} ({round(room_pred[probs_indices_sorted[i]], 3)})' for i in range(3)]
    for i, text in enumerate(pred_text):
        cv2.putText(img=blended_im, text=text, org=(50, plan.shape[0] - 100 + (i * 35)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)

    global CURRENT_ROOM, DIFF_ROOM_COUNTER
    kamer = vertices[probs_indices_sorted[0]]
    
    # TODO: Dit kijkt als er 10x iets anders dan de huidige kamer voorspeld is.
    # Als je 9x dezelfde voorspelling krijgt en dan voor de 10e een andere dan
    # wordt de nieuwe kamer die dat slechts 1 keer gedetecteerd werd?
    #
    # Misschien beter om een frequentielijst bij te houden en pas de kamer te wisselen
    # als dezelfde kamer 10x de hoogste probabiliteit heeft.
    if CURRENT_ROOM != kamer:
        DIFF_ROOM_COUNTER += 1
        if DIFF_ROOM_COUNTER > 10:
            CURRENT_ROOM = kamer
            DIFF_ROOM_COUNTER = 0

            visited_rooms.append(room_center_coords[kamer])

    for i in range(len(visited_rooms) - 2):
        cv2.line(blended_im, visited_rooms[i], visited_rooms[i+1], (255,0,0), 2, cv2.LINE_AA)

    if(len(visited_rooms) > 1):
        cv2.arrowedLine(blended_im, visited_rooms[len(visited_rooms) - 2], visited_rooms[len(visited_rooms)- 1], (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('HMM Visualization', blended_im)

def main():
    if len(sys.argv) != 7:
        raise ValueError('Provide paths to the video, calibration file and the database file')
    
    # CLI arguments
    # TODO: use parser
    video_path = sys.argv[1]
    calibration_file = sys.argv[2]
    database_file = sys.argv[3]
    csv_path = sys.argv[4]
    map_path = sys.argv[5]
    map_contour_file = sys.argv[6]

    is_gopro = False

    # Matching mode
    mode = Mode.ORB.value
    #mode = Mode.FVECTOR.value
    #mode = Mode.COMBINATION.value

    # Video setup and properties
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # fastforward in video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create pipeline instances
    preproc = FrameProcessor(calibration_file, (width, height))
    detector = PaintingDetector()
    matcher = PaintingMatcher(csv_path, database_file,features=100,mode=mode)
    localiser = Localiser(matcher=matcher, hmm_distribution='gaussian')

    # For map visualization
    map_img = cv2.imread(map_path)
    visited_rooms = []

    cv2.namedWindow('Video')

    while True:
        success, img = cap.read()

        print(CURRENT_ROOM, DIFF_ROOM_COUNTER, visited_rooms)

        if not success:
            break

        # Pass frame to processing pipeline if sharpness metric is within bounds.
        is_blurred = FrameProcessor.sharpness_metric(img, print_metric=False)
        if is_blurred:
            cv2.imshow('Video', resize_with_aspectratio(img, width=500))
        else:
            # For videos taken with GoPro camera
            if is_gopro: 
                img = preproc.undistort(img)

            detector.img = img
            contour_results, img_with_contours = detector.contours(display=False)

            room_prediction = localiser.localise(img, contour_results, display=False)
            cv2.imshow('Video', img_with_contours)

            # Visualize output of the hidden markov model.
            create_map(localiser.prob_array, map_img.copy() , map_contour_file, visited_rooms)

        k = cv2.waitKey(int(1000 / fps / 1.5))
        if k != -1:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()