# computervisie-group8
https://www.overleaf.com/8694155824wvwpcnkrcxzz


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

function benchmarkmatching {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv 'src/data/keypoints.csv' \
        --basefolder 'data/Database' \
        --what 'matcher' \
        --out 'src/data/matchingscores.csv' \
        --display ${display}
}

function generatekeypoints {

    path_test_image='src/data/test_images/Screenshot 2022-04-20 at 21.23.44.png'
    directory_database='data/Database'
    csv_path='src/data/keypoints.csv'

    python src/matcher.py "${path_test_image}" ${directory_database} ${csv_path}
}

```
