EXPERIMENT_ID="${PBS_ARRAY_INDEX:-0}"

/usr/local/bin/singularity exec --nv -B /home/Shared-AVA/AVA-dataset:/home/Shared-AVA/AVA-dataset:ro ~/docker_images/tensorflow.sif python $file $EXPERIMENT_ID $experiments