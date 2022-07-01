#! /bin/sh

EXPERIMENT_FILE="$AQA_AUGMENT_EXPERIMENTS_PATH/$2"
# Count the number of lines from the YAML file
JOBS=$(awk '/^\s\s-/ {count += 1} END {print count}' $EXPERIMENT_FILE)
JOBS=$(($JOBS-1))

# Variables for qsub
VARS="file=$AQA_AUGMENT_ROOT/$1,experiments=$EXPERIMENT_FILE,AVA_cache,AVA_images_folder,AVA_info_folder"

# Run qsub
if [ $JOBS -lt 2 ]; then
        qsub -l select=ngpus=1:mem=8gb -v $VARS cluster_runner.sh
else
        qsub -l select=ngpus=1:mem=8gb -J 0-$JOBS -v $VARS cluster_runner.sh
fi