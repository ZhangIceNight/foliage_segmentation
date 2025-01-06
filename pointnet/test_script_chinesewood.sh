# export CUDA_VISIBLE_DEVICES=1

# if [ -z "$1" ]; then
#     echo "错误：请指定存储目录编号"
#     echo "使用方法: sh train_script.sh <编号>"
#     exit 1
# fi

# log_dir="log/sem_seg_chinesewood/pt_hmamba_chinesewood_n4096_g64_k4_0"
log_dir="log/sem_seg_chinesewood/pointnet2_leaf_seg_chinesewood_0"

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

# pointnet2
python test_chinesewood.py \
    --model pointnet2_sem_seg \
    --batch_size 1 \
    --npoint 4096 \
    --log_dir $log_dir \
    --gpu 0 \
    --ckpts "$log_dir/checkpoints/best_model.pth"

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

# # pt_hmamba
# python test_chinesewood.py \
#     --model pt_hmamba \
#     --batch_size 1 \
#     --npoint 4096 \
#     --log_dir $log_dir \
#     --gpu 0 \
#     --ckpts "$log_dir/checkpoints/best_model.pth" 



# #pt_mamba_random   
# python train_fs.py \
#     --model pt_mamba_random \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 100 \
#     --learning_rate 0.0002 \
#     --optimizer AdamW \
#     --log_dir "pt_mamba_random_$1" \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

