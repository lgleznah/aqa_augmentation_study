EXPERIMENT_ID="${PBS_ARRAY_INDEX:-0}"
cd $AQA_AUGMENT_ROOT

/usr/local/bin/singularity exec --nv ~/docker_images/tensorflow.sif python $file $EXPERIMENT_ID $experiments