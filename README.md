# computervisie-group8

[Paper (local)](./paper/Computervisie_paper.pdf) or https://www.overleaf.com/8694155824wvwpcnkrcxzz

# Powerpoint

[Presentation](./docs/Group8_CV_project_presentation.pdf)

# Code overview

- **main.py** contains the main control loop of the program and visualizes the state of the hidden markov model.
- **preprocessing.py** defines the wavelet based sharpness metric and the code to calibrate a camera or load a calibration file.
- **detector.py** contains the unsupervised detection pipeline.
- **matcher.py** contains all the logic to match paintings based on the feature vector representation and the detected ORB keypoints.
- **localiser.py** and **hmm.py** combine the results of the detector and matcher to predect the current location using a hidden markov model.
- **util.py** and **graph.py** are general utilities used throughout the code, the graph class is mainly used in the localization part.
- **benmark.py**, **benchmark_fvector_matching.ipynb** and **benchmark_keypoint_matching** contain the benchmarking code for the detector and the matcher.

Most files that are the base of the pipeline (detector, matcher, localizer) contain a seperate main method to run them as individual components with self inserted parameters. This was used for testing.


# Install requirements

```bash
 pip3 install -r requirements.txt
```

# Dummy commands

Generate keypoints:

```bash
./Taskfile generatekeypoints  
```

Execute main program:

```bash
./Taskfile start  
```


# Sample usage using a Taskfile (Robbe Ubuntu, absolute paths)
**TODO:update**
```bash
#!/bin/bash

# Usage: ./Taskfile <fname>

function start {
    # Runs the main.py file with a selected video
    calib_file='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-M.npy'
    directory_database='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    csv_path='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/keypoints.csv'
    map_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/groundplan_msk.PNG'
    map_contour_file='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/polygons.npy'

    if [ $# -eq 0 ]; then
        # Video taken with a smartphone camera
        vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/smartphone/MSK_03.mp4'
        # Gopro video
        # vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/MSK_15.mp4'

        echo "No arguments supplied, playing default video @ ${vid_path}"
        python3 src/main.py ${vid_path} ${calib_file} ${directory_database} ${csv_path} ${map_path} ${map_contour_file}
    else
        python3 src/main.py "$1" ${calib_file} ${directory_database} ${csv_path} ${map_path} ${map_contour_file}
    fi

}

function detector {
    python3 src/detector.py '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/test_pictures_msk/20190217_102511.jpg'
    #python3 src/detector.py '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/dataset_pictures_msk/Zaal_A/20190323_111327.jpg'
}

function benchmark {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_log.csv' \
        --basefolder '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/dataset_pictures_msk' \
        --out '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/detectionproblems.csv' \
        --display ${display} \
        --what 'matcher'
}

function matcher {
    path_test_image='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/test_images/Screenshot 2022-04-20 at 21.23.44.png'
    directory_database='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    csv_path='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/keypoints.csv'

    python3 src/matcher.py "${path_test_image}" ${directory_database} ${csv_path}
}

function preproc {
    gopro_video='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/MSK_15.mp4'
    calib_file='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-M.npy'

    python3 src/preprocessing.py "${gopro_video}" "${calib_file}"
}

function local() {
    vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/smartphone/MSK_03.mp4'
    #vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/smartphone/MSK_08.mp4'
    directory_database='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    csv_path='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/keypoints.csv'

    python3 src/test.py ${vid_path} ${directory_database} ${csv_path}
}

"$@"
```


# Sample usage using a Taskfile (Lennert macOS Monterey, relative paths)

```bash
#!/bin/bash

# Usage: ./Taskfile <function_name>

function start {
    # Runs the main.py file with a selected video
    calib_file='src/data/gopro-M.npy'
    directory_database='data/Database'
    csv_path='src/data/keypoints.csv'
    map_path='data/groundplan_msk.png'
    map_contour_file='src/data/polygons.npy'

    if [ $# -eq 0 ]; then
        # Video taken with a smartphone camera
        # vid_path='data/videos/smartphone/MSK_03.mp4'
        vid_path='data/videos/smartphone/MSK_05.mp4'
        # Gopro video
        # vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/MSK_15.mp4'

        echo "No arguments supplied, playing default video @ ${vid_path}"
        python src/main.py ${vid_path} ${calib_file} ${directory_database} ${csv_path} ${map_path} ${map_contour_file}
    else
        python src/main.py "$1" ${calib_file} ${directory_database} ${csv_path} ${map_path} ${map_contour_file}
    fi

}


function benchmark {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv 'data/Database_log.csv' \
        --basefolder 'data/Computervisie 2020 Project Database/dataset_pictures_msk' \
        --out 'src/csv/detectionproblems.csv' \
        --display ${display}
}

function benchmarkmatchingkeypoints {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv 'src/data/keypoints.csv' \
        --basefolder 'data/Database' \
        --what 'matcherkeypoints' \
        --out 'src/data/matchingscores_keypoints.csv' \
        --display ${display}
}

function benchmarkmatchingfvector {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv 'src/data/keypoints.csv' \
        --basefolder 'data/Database' \
        --what 'matcherfvector' \
        --out 'src/data/matchingscores_fvector.csv' \
        --display ${display}
}


function generatekeypoints {

    path_test_image='src/data/test_images/Screenshot 2022-04-20 at 21.23.44.png'
    directory_database='data/Database'
    csv_path='src/data/keypoints.csv'

    python src/matcher.py "${path_test_image}" ${directory_database} ${csv_path}
}


"$@"

```
