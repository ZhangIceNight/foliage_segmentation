# export CUDA_VISIBLE_DEVICES=1

# if [ -z "$1" ]; then
#     echo "错误：请指定存储目录编号"
#     echo "使用方法: sh train_script.sh <编号>"
#     exit 1
# fi

# log_dir="log/sem_seg/pt_hmamba_N2_0"
log_dir="log/sem_seg/pct_seg_0"



# pointnet
# python train_fs.py \
#     --model pointnet_sem_seg \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.001 \
#     --optimizer Adam \
#     --log_dir "pointnet_leaf_seg_$1" \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

# # pointnet2
# python train_fs.py \
#     --model pointnet2_sem_seg \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.001 \
#     --optimizer Adam \
#     --log_dir "pointnet2_leaf_seg_$1" \
#     --gpu 0 

# # pt_mamba
# python train_fs.py \
#     --model pt_mamba \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 300 \
#     --learning_rate 0.0002 \
#     --optimizer AdamW \
#     --log_dir "pt_mamba_$1" \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

# pt_hmamba
# python test_pc.py \
#     --model pt_hmamba \
#     --batch_size 1 \
#     --npoint 4096 \
#     --log_dir $log_dir \
#     --gpu 0 \
#     --ckpts "$log_dir/checkpoints/best_model.pth" 



# pct_seg
python test_pc.py \
    --model pct_seg \
    --batch_size 1 \
    --npoint 40960 \
    --log_dir $log_dir \
    --gpu 0 \
    --ckpts "$log_dir/checkpoints/best_model.pth" 
