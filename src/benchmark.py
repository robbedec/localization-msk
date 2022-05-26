from turtle import distance
import pandas as pd
import numpy as np
import os
import cv2
import argparse
import matplotlib.pyplot as plt
import time

from shapely.geometry import Polygon
import json

from detector import PaintingDetector
from matcher import PaintingMatcher
from matcher import Distance
from matcher import Mode
from util import printProgressBar, rectify_contour

"""
Usage:
    python3 src/benchmark.py \
        --csv '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_log.csv' \
        --basefolder '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/dataset_pictures_msk' \
        --out '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/detectionproblems.csv' \
        --display 'y' \
        --what 'all'
"""

parser = argparse.ArgumentParser(description="My Script")
parser.add_argument('--csv', help='Path to master CSV', required=True, type=str)
parser.add_argument('--basefolder', help='Path to the base folder that contains the images', required=True, type=str)
parser.add_argument('--out', help='Path to store the output csv', required=True, type=str)
parser.add_argument('--display', help='Display intermediate images', required=False, default='y', type=str)
parser.add_argument('--what', help='Which benchmark to run: all|detector|matcher', required=True, type=str)

args = vars(parser.parse_args())
CSV_PATH = args['csv']
IMAGES_PATH = args['basefolder']
OUT_PATH = args['out']
display = args['display'] == 'y'
what = args['what']

# Calculate the intersection over union ratio for two bouning boxes
def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def string_to_array(val):
  return list(map(lambda x: int(x.strip()), val[1: -1].split(',')))

def benchmark_detector():
    print('---------------------------------------------')
    print('BENCHMARKING PAINTING DETECTOR')
    print('---------------------------------------------')

    detector = PaintingDetector(bbox_color=(0, 0, 255))

    IOU_scores = []
    false_negatives = 0
    false_positives = 0

    df_paintings = pd.read_csv(CSV_PATH)
    df_detection_problems = pd.DataFrame()

    # Create image path
    df_paintings['image_path'] = df_paintings.apply(lambda row: os.path.join(IMAGES_PATH, row['Room'], row['Photo'] + '.jpg'), axis=1)

    for impath, df_group in df_paintings.groupby('image_path'):
        current_false_positives = 0
        print(impath)

        # Feed image to the detector and calculate painting locations.
        img = cv2.imread(impath)
        (old_h, old_w, _) = img.shape

        detector.img = cv2.imread(impath)
        res, img_with_contours = detector.contours()

        (new_h, new_w, _) = img_with_contours.shape
        scaleY, scaleX = old_h / new_h, old_w / new_w
        
        # Contains all ground truth boxes from the current image.
        ground_truth = []
        for index, row in df_group.iterrows():
            ground_truth_bbox = np.array([
                string_to_array(row['Top-left']),
                string_to_array(row['Top-right']),
                string_to_array(row['Bottom-right']),
                string_to_array(row['Bottom-left'])
            ])

            ground_truth.append(ground_truth_bbox)

            # Coordinates are scaled to the downsized detector image.
            # This is only done for visualization purposes.
            # The ground truth box is shown in green, the detected box in red.
            gt_bbox_rescaled = np.array([ np.apply_along_axis(lambda row: np.rint(np.divide(row, [scaleX, scaleY])).astype(int), 0, c) for c in ground_truth_bbox])
            cv2.drawContours(img_with_contours, [gt_bbox_rescaled], 0, (0, 255, 0), 2, cv2.LINE_8)

        for prediction_index in range(res.shape[0]):
            # Match polygon from the results to a polygon of the database.
            # Do this by checking all detected ROI's and take the one that has a value for IOU.
            # Contours that are not in this area will have an IOU = 0.
            # The highest IOU corresponds to two matching bounding boxes.
            IOUS = [ calculate_iou(gt_box, res[prediction_index]) for gt_box in ground_truth ]
            IOUS.sort(reverse=True)
            IOU = IOUS[0]

            if IOU == 0:
                # Detection is not a painting => False positive
                print('FP: Detection is not a painting')
                false_positives += 1
                current_false_positives += 1

                df_detection_problems = df_detection_problems.append({
                    'Room': df_group.iloc[0]['Room'],
                    'Photo': df_group.iloc[0]['Photo'],
                    'path': impath,
                    'kind': 'FP',
                }, ignore_index=True)

                if display:
                    cv2.imshow('FP: Detection is not a painting', img_with_contours)
                    k = cv2.waitKey(0)
                    cv2.destroyWindow('FP: Detection is not a painting')
                    if k == 27:    # Esc key to stop
                        break

                continue

            IOU_scores.append(IOU)
            print(IOU)

            df_detection_problems = df_detection_problems.append({
                'Room': df_group.iloc[0]['Room'],
                'Photo': df_group.iloc[0]['Photo'],
                'path': impath,
                'kind': 'TP',
            }, ignore_index=True)

        # Calculate false negatives for current group.
        current_false_negatives = len(df_group) - (res.shape[0] - current_false_positives)
        false_negatives += current_false_negatives
        for i in range(current_false_negatives):
            print('FN: Painting not detected')
            df_detection_problems = df_detection_problems.append({
                'Room': df_group.iloc[0]['Room'],
                'Photo': df_group.iloc[0]['Photo'],
                'path': impath,
                'kind': 'FN',
            }, ignore_index=True)

        if display and current_false_negatives > 0:
            cv2.imshow('FN: Painting not detected', img_with_contours)
            k = cv2.waitKey(0)
            cv2.destroyWindow('FN: Painting not detected')
            if k == 27:    # Esc key to stop
                break

            continue

        if display:
            cv2.imshow('Detected', img_with_contours)
            k = cv2.waitKey(0)
            cv2.destroyWindow('Detected')
            if k == 27:    # Esc key to stop
                break

    print('Avergage intersection over union score: {}\n \
        Total amount of paintings: {}\n \
        Paintings detected: {}\n \
        False positives: {}\n \
        False negatives: {}' \
        .format(sum(IOU_scores) / len(IOU_scores), len(df_paintings), len(IOU_scores), false_positives, false_negatives))

    # CSV file can be used to show images that casue problem. Feed those to
    # the detector with display option True to visualize the internal images.
    if os.path.exists(OUT_PATH):
        if input('Overwrite file? [yes|y|no|n] ').lower() in ['yes', 'y']:
            df_detection_problems.to_csv(OUT_PATH)
    else:
        df_detection_problems.to_csv(OUT_PATH)

    # Distribution of IOC scores shown over buckets with size 10%.
    plt.hist(IOU_scores, bins=np.linspace(0, 1, 11), ec='black')
    plt.xticks(np.linspace(0, 1, 11))
    plt.title('Distribution of IOU scores for the detected paintings.')
    plt.xlabel('Intersection over Union score')
    plt.ylabel('Amount')

    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUT_PATH)), 'benchmark_images', 'IOU_distribution.jpg'))
    plt.clf()

    # Distribution of prediction errors by museum hall number (including true positives).
    df_problems_grouped_by_hall = df_detection_problems.groupby(['Room', 'kind'])['Photo'].count()

    ax = (df_detection_problems.drop(['path'], axis=1).groupby(['Room','kind']).count().unstack('kind').plot.bar(figsize=(11, 7)))
    ax.legend(['False negatives', 'False positives', 'True positives'])

    plt.ylabel('Amount of false negatives / positives')
    plt.title('Anomalies paintings grouped by hall number')
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUT_PATH)), 'benchmark_images', 'grouped_by_hall_include_TP.jpg'))
    plt.clf()

    # Distribution of prediction errors by museum hall number (excluding true positives).
    df_problems_grouped_by_hall = df_detection_problems.groupby(['Room', 'kind'])['Photo'].count()

    ax = (df_detection_problems[df_detection_problems['kind'] != 'TP'].drop(['path'], axis=1).groupby(['Room','kind']).count().unstack('kind').plot.bar(figsize=(11, 7)))
    ax.legend(['False negatives', 'False positives'])

    plt.ylabel('Amount of false negatives / positives')
    plt.title('Anomalies paintings grouped by hall number')
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUT_PATH)), 'benchmark_images', 'grouped_by_hall.jpg'))

def benchmark_matcher_keypoints():
    print('---------------------------------------------')
    print('BENCHMARKING PAINTING KEYPOINTS')
    print('---------------------------------------------')

    df = pd.DataFrame(columns=['filename', 'result_50_features', 'distance_50_features', 
                                 'second_result_50_features', 'second_distance_50_features', 'time_50_features',
                                 'result_100_features', 'distance_100_features',
                                 'second_result_100_features', 'second_distance_100_features', 'time_100_features',
                                 'result_200_features', 'distance_200_features',
                                 'second_result_200_features', 'second_distance_200_features', 'time_200_features',
                                 'result_300_features', 'distance_300_features',
                                 'second_result_300_features', 'second_distance_300_features', 'time_300_features', 
                                 'result_fvector', 'distance_fvector','second_result_fvector', 'second_distance_fvector', 'time_fvector'])


    print('----------------')
    print('50 FEATURES')
    print('----------------')

    features  =  50
    df = match_number_of_features(features, df, 'result_50_features', 'distance_50_features', 'time_50_features',True)

    print('----------------')
    print('100 FEATURES')
    print('----------------')
    features  = 100
    match_number_of_features(features, df, 'result_100_features', 'distance_100_features', 'time_100_features')

    print('----------------')
    print('200 FEATURES')
    print('----------------')
    features  = 200
    match_number_of_features(features, df,'result_200_features', 'distance_200_features', 'time_200_features')

    print('----------------')
    print('300 FEATURES')
    print('----------------')
    features  = 300
    match_number_of_features(features, df, 'result_300_features', 'distance_300_features', 'time_300_features')


    df.to_csv(OUT_PATH)  


def match_number_of_features(features, df, col_filename, col_distance, col_time, overwrite=False):
    # Generate database file (for a particular amount of features)
    PaintingMatcher.generate_keypoints(IMAGES_PATH,CSV_PATH, features, fvector_state=overwrite)

    # Set-up matcher and detector
    detector = PaintingDetector()
    matcher = PaintingMatcher(CSV_PATH,IMAGES_PATH,features,mode=Mode.FVECTOR_EUCLIDEAN) # Force fvector load :D

    directory_list = "/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/data/Computervisie 2020 Project Database/dataset_pictures_msk"

    progress_dic = 0
    progress = 0
    printProgressBar(progress_dic, len(os.listdir(directory_list)), prefix = 'Progress matching:', suffix = 'Complete', length = 50)

    # Loop through directory which has subfolders
    for file in os.listdir(directory_list):
        directory = os.fsdecode(file)
        sub_dir_path = directory_list + '/' + directory

        if (os.path.isdir(sub_dir_path)):

            for image_name in os.listdir(sub_dir_path):
                filename = os.fsdecode(image_name)

                # Load image
                img_path = sub_dir_path + '/' + filename
                img = cv2.imread(img_path)

                # Pass image through detector
                detector.img = img
                contour_results, img_with_contours = detector.contours(display=False)


                # ORB MATCHING
                filename_match = []
                distance = []
                filename_match_second = []
                distance_second = []
                timing = []

                # Fvector matching
                filename_match_fv = []
                distance_fv = []
                filename_match_second_fv = []
                distance_second_fv = []
                timing_fv = []


                # Add row when filename is not in dataframe
                if(not (filename in df['filename'].unique())):
                    df.loc[progress] = [filename, None, None, None, None, None, None, None, None, None, None, None, None,None,None,None,None,None,None,None,None,None,None,None,None,None]

                # Loop through all detected boxes
                for contour in contour_results:

                    matcher.mode = Mode.ORB # Activate ORB matcher

                    # ORB benchmark
                    affine_image,crop_img = rectify_contour(contour, img, display=False)

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img)
                    toc = time.perf_counter()


                    if len(distances) > 0:
                        filename_match.append(matcher.get_filename(distances[0][0]))
                        distance.append(distances[0][1])

                    if len(distances) > 1:
                        filename_match_second.append(matcher.get_filename(distances[1][0]))
                        distance_second.append(distances[1][1])

                    timing.append(toc-tic)

                    if(overwrite == True):
                        
                        matcher.mode = Mode.FVECTOR_EUCLIDEAN # Activate fvector matcher

                        # Fvector  benchmark
                        tic = time.perf_counter()
                        distances = matcher.match(crop_img)
                        toc = time.perf_counter()

                        if len(distances) > 0:
                            filename_match_fv.append(matcher.get_filename(distances[0][0]))
                            distance_fv.append(distances[0][1])

                        if len(distances) > 1:
                            filename_match_second_fv.append(matcher.get_filename(distances[1][0]))
                            distance_second_fv.append(distances[1][1])

                        timing_fv.append(toc-tic)


                indexes = df.index[df['filename'] == filename].tolist()   

                df.at[indexes[0], col_filename] =  json.dumps(filename_match)
                df.at[indexes[0], col_distance] =  json.dumps(distance)
                df.at[indexes[0], "second_" + col_filename] =  json.dumps(filename_match_second)
                df.at[indexes[0], "second_" + col_distance] =  json.dumps(distance_second)
                df.at[indexes[0], col_time] =  json.dumps(timing)

                if(overwrite == True):
                    df.at[indexes[0], 'result_fvector'] =  json.dumps(filename_match_fv)
                    df.at[indexes[0], 'distance_fvector'] =  json.dumps(distance_fv)
                    df.at[indexes[0], 'second_result_fvector'] =  json.dumps(filename_match_second_fv)
                    df.at[indexes[0], 'second_distance_fvector'] =  json.dumps(distance_second_fv)
                    df.at[indexes[0], 'time_fvector'] =  json.dumps(timing_fv)

                
                progress += 1

            progress_dic += 1    
            printProgressBar(progress_dic, len(os.listdir(directory_list)), prefix = 'Progress matching:', suffix = 'Complete', length = 50)

    return df


def benchmark_matcher_vector():
    print('---------------------------------------------')
    print('BENCHMARKING PAINTING VECTOR')
    print('---------------------------------------------')

    df = pd.DataFrame(columns=['filename', 'result_euclidean', 'distance_euclidean', 'second_distance_euclidean','time_euclidean',
                            'result_cityblock','distance_cityblock','second_distance_cityblock','time_cityblock',
                            'result_minkowski','distance_minkowski','second_distance_minkowski','time_minkowski',
                            'result_chebyshev','distance_chebyshev','second_distance_chebyshev','time_chebyshev',
                            'result_cosine','distance_cosine','second_distance_cosine','time_cosine',
                            'result_jaccard','distance_jaccard','second_distance_jaccard','time_jaccard'])



    # Set-up matcher and detector
    detector = PaintingDetector()
    matcher = PaintingMatcher(CSV_PATH,IMAGES_PATH,100,mode=Mode.FVECTOR) # Force fvector load :D
  

    directory_list = "/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/data/Computervisie 2020 Project Database/dataset_pictures_msk"


    progress_dic = 0
    progress = 0
    printProgressBar(progress_dic, len(os.listdir(directory_list)), prefix = 'Progress matching:', suffix = 'Complete', length = 50)

    # Loop through directory which has subfolders
    for file in os.listdir(directory_list):
        directory = os.fsdecode(file)
        sub_dir_path = directory_list + '/' + directory

        if (os.path.isdir(sub_dir_path)):

            for image_name in os.listdir(sub_dir_path):
                filename = os.fsdecode(image_name)

                # Load image
                img_path = sub_dir_path + '/' + filename
                img = cv2.imread(img_path)

                # Pass image through detector
                detector.img = img
                contour_results, img_with_contours = detector.contours(display=False)

                
                filename_match_euclidean = []
                distance_euclidean = []
                distance_second_euclidean = []
                timing_euclidean = []

                filename_match_cityblock = []
                distance_cityblock = []
                distance_second_cityblock = []
                timing_cityblock = []

                filename_match_minkowski = []
                distance_minkowski = []
                distance_second_minkowski = []
                timing_minkowski = []


                filename_match_chebyshev = []
                distance_chebyshev = []
                distance_second_chebyshev = []
                timing_chebyshev = []

                filename_match_cosine = []
                distance_cosine = []
                distance_second_cosine = []
                timing_cosine = []

                filename_match_jaccard = []
                distance_jaccard = []
                distance_second_jaccard = []
                timing_jaccard = []

                # Loop through all detected boxes
                for contour in contour_results:

                    # Rectify
                    affine_image,crop_img = rectify_contour(contour, img, display=False)


                    # Euclidean

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.EUCLIDEAN)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_euclidean.append(matcher.get_filename(distances[0][0]))
                        distance_euclidean.append(distances[0][1])

                    if len(distances) > 1:
                        distance_second_euclidean.append(distances[1][1])

                    timing_euclidean.append(float(toc-tic))

                    # Cityblock

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.CITYBLOCK)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_cityblock.append(matcher.get_filename(distances[0][0]))
                        distance_cityblock.append(float(distances[0][1]))

                    if len(distances) > 1:
                        distance_second_cityblock.append(float(distances[1][1]))

                    timing_cityblock.append(float(toc-tic))

                    # Minkowski

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.MINOWSKI)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_minkowski.append(matcher.get_filename(distances[0][0]))
                        distance_minkowski.append(distances[0][1])

                    if len(distances) > 1:
                        distance_second_minkowski.append(distances[1][1])

                    timing_minkowski.append(float(toc-tic))


                    # Chebyshev

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.CHEBYSHEV)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_chebyshev.append(matcher.get_filename(distances[0][0]))
                        distance_chebyshev.append(float(distances[0][1]))

                    if len(distances) > 1:
                        distance_second_chebyshev.append(float(distances[1][1]))

                    timing_chebyshev.append(float(toc-tic))

                    # Cosine

                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.COSINE)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_cosine.append(matcher.get_filename(distances[0][0]))
                        distance_cosine.append(distances[0][1])

                    if len(distances) > 1:
                        distance_second_cosine.append(distances[1][1])

                    timing_cosine.append(float(toc-tic))

                    # Jacard
                    tic = time.perf_counter()
                    distances = matcher.match(crop_img,dist_metric=Distance.JACCARD)
                    toc = time.perf_counter()

                    if len(distances) > 0:
                        filename_match_jaccard.append(matcher.get_filename(distances[0][0]))
                        distance_jaccard.append(distances[0][1])

                    if len(distances) > 1:
                        distance_second_jaccard.append(distances[1][1])

                    timing_jaccard.append(float(toc-tic))



                df.loc[progress] = [filename, json.dumps(filename_match_euclidean), json.dumps(distance_euclidean), json.dumps(distance_second_euclidean), json.dumps(timing_euclidean),
                                     json.dumps(filename_match_cityblock), json.dumps(distance_cityblock), json.dumps(distance_second_cityblock), json.dumps(timing_cityblock),
                                     json.dumps(filename_match_minkowski), json.dumps(distance_minkowski), json.dumps(distance_second_minkowski), json.dumps(timing_minkowski),
                                     json.dumps(filename_match_chebyshev), json.dumps(distance_chebyshev), json.dumps(distance_second_chebyshev), json.dumps(timing_chebyshev),
                                     json.dumps(filename_match_cosine), json.dumps(distance_cosine), json.dumps(distance_second_cosine), json.dumps(timing_cosine),
                                     json.dumps(filename_match_jaccard), json.dumps(distance_jaccard), json.dumps(distance_second_jaccard), json.dumps(timing_jaccard)]

                progress += 1

            progress_dic += 1    
            printProgressBar(progress_dic, len(os.listdir(directory_list)), prefix = 'Progress matching:', suffix = 'Complete', length = 50)


    df.to_csv(OUT_PATH)  

# SETUP:
if what == 'all':
    benchmark_detector()
    benchmark_matcher_keypoints()
    benchmark_matcher_vector()
elif what == 'matcherkeypoints':
    benchmark_matcher_keypoints()
elif what == 'matcherfvector':
    benchmark_matcher_vector()
elif what == 'detector':
    benchmark_detector()
else:
    print('Unknown argument')
    exit()