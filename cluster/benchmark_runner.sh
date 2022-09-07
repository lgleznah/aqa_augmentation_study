cd $AQA_AUGMENT_ROOT

/usr/local/bin/singularity exec ~/docker_images/tensorflow.sif python cluster/benchmark.py $results $parallel $path