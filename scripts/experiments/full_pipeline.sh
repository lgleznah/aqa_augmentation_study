#! /bin/sh

for i; do
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh trainer.py $i true
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh predictor.py $i true
    $AQA_AUGMENT_ROOT/scripts/experiments/sequential_launcher.sh metrics.py $i true
done