#!/bin/bash

# Usage examples:
# bash scripts/train.sh --config-name ${Dataset_type}_${Model_type}_lr1e-3_bs32 --fold 5

# Default parameters
CONFIG_NAME="default"
BATCH_SIZE=""
RESUME=""
EPOCHS=""
LR=""
DEBUG=0

# parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config-name) CONFIG_NAME="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --fold) FOLD_NUM="$2"; shift ;;
        --debug) DEBUG=1 ;;  # only run on fold 0 if --debug is passed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

#  construct Hydra arguments
HYDRA_ARGS=""
[ -n "$BATCH_SIZE" ] && HYDRA_ARGS+=" data.batch_size=$BATCH_SIZE"
[ -n "$RESUME" ] && HYDRA_ARGS+=" model.resume=$RESUME"
[ -n "$EPOCHS" ] && HYDRA_ARGS+=" trainer.max_epochs=$EPOCHS"
[ -n "$LR" ] && HYDRA_ARGS+=" model.learning_rate=$LR"


# get project root dir
PROJECT_ROOT=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

if [ "$DEBUG" -eq 1 ]; then
    folds=(0)
else
    folds=($(seq 0 $((FOLD_NUM-1))))
fi


for fold in ${folds[@]}; do
    CMD_HYDRA_ARGS="$HYDRA_ARGS data.fold_idx=$fold"
    python train.py --config-name $CONFIG_NAME $CMD_HYDRA_ARGS # fold 0-4
    # python train.py --config-name $CONFIG_NAME $HYDRA_ARGS # fold 0 for 5 times used for repeat exps
done