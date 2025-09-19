#!/bin/bash

# 用法:
#   ./evaluate.sh --config-name PLU_AUT_pointnet_lr1e-3_bs32 --fold 5
#   ./evaluate.sh --config-name PLU_AUT_pointnet_lr1e-3_bs32 --fold 1
# bash scripts/evaluate.sh --config-name PLU_AUT_pointnet_lr1e-3_bs32 --fold 1 --resume path/to/ckpt.ckpt
# bash scripts/evaluate.sh --config-name ForestSemantic_Difficult_dhmamba_mse_lr2e-4_bs16 --fold 1 --resume 
CONFIG_NAME="default"
RESUME=""
FOLD_NUM=""
DEBUG=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config-name) CONFIG_NAME="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --fold) FOLD_NUM="$2"; shift ;;
        --debug) DEBUG=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 构建 Hydra 参数
HYDRA_ARGS=""
[ -n "$RESUME" ] && HYDRA_ARGS+=" model.resume=$RESUME"

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
    python evaluate.py --config-name $CONFIG_NAME $CMD_HYDRA_ARGS
done
