# computervisie-group8

# Sample usage using a Taskfile (Robbe Ubuntu, absolute paths)

```bash
#!/bin/bash

# Usage: ./Taskfile <function_name>

function start {
    # Runs the main.py file with a selected video
    if [ $# -eq 0 ]; then
        vid_path='/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/smartphone/MSK_03.mp4'

        echo "No arguments supplied, playing default video @ ${vid_path}"
        python3 src/main.py ${vid_path}
    else
        python3 src/main.py "$1"
    fi

}

function detector {
    python3 src/detector.py
}

function benchmark {
    display='y'
    if ! [ $# -eq 0 ]; then
        display="$1"
    fi

    python3 src/benchmark.py \
        --csv '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_log.csv' \
        --basefolder '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/dataset_pictures_msk' \
        --out '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/csv/detectionproblems.csv' \
        --display ${display}
}

"$@"
```


# Sample usage using a Taskfile (Lennert macOS Moterey, relative paths)

```bash
#!/bin/bash

# Usage: ./Taskfile <function_name>


# Start: ./Taskfile start
# Start: ./Taskfile start data/videos/smartphone/MSK_03.mp4

function start {
    # Runs the main.py file with a selected video
    if [ $# -eq 0 ]; then
        vid_path='data/videos/smartphone/MSK_03.mp4'

        echo "No arguments supplied, playing default video @ ${vid_path}"
        python3 src/main.py ${vid_path}
    else
        python3 src/main.py "$1"
    fi

}


# Detector: ./Taskfile detector

function detector {
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/test_pictures_msk/20190217_102133.jpg'
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/test_pictures_msk/20190203_110338.jpg'
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/test_pictures_msk/20190217_110614.jpg'
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/test_pictures_msk/20190217_102014.jpg'
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/test_pictures_msk/20190217_102511.jpg'
    python3 src/detector.py  'data/Computervisie 2020 Project Database/dataset_pictures_msk/Zaal_A/20190323_111313.jpg'
    # python3 src/detector.py  'data/Computervisie 2020 Project Database/dataset_pictures_msk/zaal_1/IMG_20190323_111739.jpg'
}


# Benchmark: ./Taskfile benchmark 0
# Benchmark without displaying results: ./Taskfile benchmark 0

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

"$@"
```