# computervisie-group8

# Sample usage using a Taskfile

```bash
#!/bin/bash

# Usage: ./Taskfile <fname>

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
