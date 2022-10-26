#! /bin/sh

EXPERIMENT_FILE="$AQA_AUGMENT_EXPERIMENTS_PATH/$2"
# Count the number of lines from the YAML file
JOBS=$(grep 'name' $EXPERIMENT_FILE | wc -l)
JOBS_ITER=$(($JOBS-1))

RERUN=$3

echo "Found ${JOBS} experiments to run!"

FILE=$AQA_AUGMENT_ROOT/$1

for i in $(seq 0 $JOBS_ITER)
do
    echo "Running experiment with index ${i}"
    python $FILE $i $EXPERIMENT_FILE $RERUN
done