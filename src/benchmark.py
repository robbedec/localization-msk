import pandas as pd
import numpy as np
import os
import cv2
import argparse

from shapely.geometry import Polygon

from detector import PaintingDetector

"""
Usage:
    python3 src/benchmark.py \
        --csv '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_log.csv' \
        --basefolder '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/dataset_pictures_msk' \
        --out '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/csv/detectionproblems.csv' \
        --display 'y'
"""

parser = argparse.ArgumentParser(description="My Script")
parser.add_argument('--csv', help='Path to master CSV', required=True)
parser.add_argument('--basefolder', help='Path to the base folder that contains the images', required=True)
parser.add_argument('--out', help='Path to store the output csv', required=True)
parser.add_argument('--display', help='Display intermediate images', required=False, default='y')

# Calculate the intersection over union ratio for two bouning boxes
def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def string_to_array(val):
  return list(map(lambda x: int(x.strip()), val[1: -1].split(',')))

args = vars(parser.parse_args())
CSV_PATH = args['csv']
IMAGES_PATH = args['basefolder']
OUT_PATH = args['out']
display = args['display'] == 'y'

detector = PaintingDetector(bbox_color=(0, 0, 255))

IOU_scores = []
false_negatives = 0
false_positives = 0

df_paintings = pd.read_csv(CSV_PATH)
df_detection_problems = pd.DataFrame()

# Create image path
df_paintings['image_path'] = df_paintings.apply(lambda row: os.path.join(IMAGES_PATH, row['Room'], row['Photo'] + '.jpg'), axis=1)

for impath, df_group in df_paintings.groupby('image_path'):
    # Feed image to the detector and calculate painting locations.
    img = cv2.imread(impath)
    (old_h, old_w, _) = img.shape

    detector.img = cv2.imread(impath)
    res, img_with_contours = detector.contours()

    (new_h, new_w, _) = img_with_contours.shape
    scaleY, scaleX = old_h / new_h, old_w / new_w
    

    # Apply scaling correction to the results. The result is a 3D tensor where the first index is a counter, the other two correspond to
    # the 4x2 representation of 4 2D points.
    # Coordinates are scaled to the size of the original image.
    res_rescaled = np.array([ np.apply_along_axis(lambda row: np.rint(np.multiply(row, [scaleX, scaleY])).astype(int), 1, c) for c in res])

    if res_rescaled.shape[0] == 0:
        print('Nothing detected')
        false_negatives += len(df_group)
        df_detection_problems.append({ 'path': impath }, ignore_index=True)

        if display:
            cv2.imshow('Not detected', img_with_contours)
            k = cv2.waitKey(0)
            if k == 27:    # Esc key to stop
                break
            cv2.destroyWindow('Not detected')

        continue

    for index, row in df_group.iterrows():
        ground_truth_bbox = np.array([
            string_to_array(row['Top-left']),
            string_to_array(row['Top-right']),
            string_to_array(row['Bottom-right']),
            string_to_array(row['Bottom-left'])
        ])

        # Coordinates are scaled to the downsized detector image.
        # This is only done for visualization purposes.
        # The ground truth box is shown in green, the detected box in red.
        gt_bbox_rescaled = np.array([ np.apply_along_axis(lambda row: np.rint(np.divide(row, [scaleX, scaleY])).astype(int), 0, c) for c in ground_truth_bbox])
        cv2.drawContours(img_with_contours, [gt_bbox_rescaled], 0, (0, 255, 0), 2, cv2.LINE_8)

        # Match polygon from the results to a polygon of the database.
        # Do this by checking all detected ROI's and take the one that has a value for IOU.
        # Contours that are not in this area will have an IOU = 0.
        # The highest IOU corresponds to two matching bounding boxes.
        IOUS = [ calculate_iou(ground_truth_bbox, res_rescaled[i]) for i in range(res_rescaled.shape[0])]
        IOUS.sort(reverse=True)
        IOU = IOUS[0]

        if IOU == 0:
            # Painting not detected.
            print('Painting not detected')
            false_negatives += 1
            df_detection_problems = df_detection_problems.append({ 'path': impath }, ignore_index=True)
            continue

        IOU_scores.append(IOU)
        print(IOU)

    if display:
        cv2.imshow('Detected', img_with_contours)
        k = cv2.waitKey(0)
        cv2.destroyWindow('Detected')
        if k == 27:    # Esc key to stop
            break

print('Avergage intersection over union score: {}\nPaintings detected: {}\nFalse negatives: {}'.format(sum(IOU_scores) / len(IOU_scores), len(IOU_scores), false_negatives))
df_detection_problems.to_csv(OUT_PATH)