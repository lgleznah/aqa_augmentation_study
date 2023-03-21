#! /bin/sh

for i; do
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh trainer.py $i false
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh predictor.py $i false
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh metrics.py $i false
done