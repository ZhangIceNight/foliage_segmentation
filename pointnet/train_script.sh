export CUDA_VISIBLE_DEVICES=0


# pointnet
# python train_semseg.py \
#     --model pointnet_sem_seg \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.001 \
#     --optimizer Adam \
#     --log_dir pointnet_leaf_seg \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

# pointnet2
# python train_semseg.py \
#     --model pointnet2_sem_seg \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.001 \
#     --optimizer Adam \
#     --log_dir pointnet2_leaf_seg_1 \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

# # pt_mamba
# python train_semseg.py \
#     --model pt_mamba \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.0002 \
#     --optimizer AdamW \
#     --log_dir pt_mamba_0 \
#     --gpu 0 \
#     --ckpts 'pretrain.pth'

# pt_hmamba
python train_semseg.py \
    --model pt_hmamba \
    --batch_size 16 \
    --npoint 4096 \
    --epoch 128 \
    --learning_rate 0.0002 \
    --optimizer AdamW \
    --log_dir pt_hmamba_0 \
    --gpu 0 \
    --ckpts 'pretrain.pth'

