from ast import Mod
import cv2
import sys
import numpy as np
import pandas as pd

from util import resize_with_aspectratio, vertices
from detector import PaintingDetector
from matcher import PaintingMatcher
from localiser import Localiser
from preprocessing import FrameProcessor
from enum import Enum


class Mode(Enum):
    ORB = 0
    FVECTOR = 1
    COMBINATION = 2

def create_map(room_pred, plan, file_path,pixel_coord):
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


    highest_prob_index = np.argmax(np.array(room_pred))
    probs_indices_sorted = np.flip(np.argsort(np.array(room_pred)))
    pred_text = [f'Zaal: {vertices[probs_indices_sorted[i]]} ({round(room_pred[probs_indices_sorted[i]], 3)})' for i in range(3)]
    for i, text in enumerate(pred_text):
        cv2.putText(img=blended_im, text=text, org=(50, plan.shape[0] - 100 + (i * 35)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)

    kamer = vertices[highest_prob_index]
    # print(kamer)
    
    if main.oude_kamer != kamer:
        main.buffer += 1
        if main.buffer > 10:
            main.oude_kamer = kamer
            main.buffer = 0
            for index, row in pixel_coord.iterrows():
                kamernummer, x, y = row
                if kamernummer == kamer:
                    punt = (x,y)
                    main.kamer_volgorde.append(punt)
    for i in range(len(main.kamer_volgorde)-2 ):
        cv2.line(blended_im, main.kamer_volgorde[i], main.kamer_volgorde[i+1], (255,0,0), 2, cv2.LINE_AA)
    if(len(main.kamer_volgorde)>1):
        cv2.arrowedLine(blended_im, main.kamer_volgorde[len(main.kamer_volgorde)-2], main.kamer_volgorde[len(main.kamer_volgorde)-1], (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('HMM Visualization', blended_im)

    # poly_fill contains the raw mask
    #cv2.imshow('poly_fill', mask)
    #cv2.waitKey(0)

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
    #mode = Mode.ORB.value
    #mode = Mode.FVECTOR.value
    mode = Mode.COMBINATION.value

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

    map_img = cv2.imread(map_path)
    pixel_coord = pd.read_csv("src/data/coord.csv")

    main.oude_kamer = 'Z'
    main.buffer = 0
    main.kamer_volgorde = []

    cv2.namedWindow('Video')

    while True:
        success, img = cap.read()

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
            #txt = "Zaal: " + room_prediction
            #cv2.putText(img=img_with_contours, text=txt, org=(50, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
            cv2.imshow('Video', img_with_contours)

            # Visualize output of the hidden markov model.
            create_map(localiser.prob_array, map_img.copy() , map_contour_file, pixel_coord)

        k = cv2.waitKey(int(1000 / fps / 1.5))
        if k != -1:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()