# qsub -l select=ngpus=1 -v results=results_NAS.txt,parallel=false,path=/home/lgleznah/src/AVA-dataset/,AQA_AUGMENT_ROOT benchmark_runner.sh

cd $AQA_AUGMENT_ROOT

/usr/local/bin/singularity exec ~/docker_images/tensorflow.sif python cluster/benchmark.py $results $parallel $path