# python train_semseg.py \
#     --model pointnet_sem_seg \
#     --batch_size 16 \
#     --npoint 4096 \
#     --epoch 128 \
#     --learning_rate 0.001 \
#     --optimizer Adam \
#     --log_dir pointnet_leaf_seg \
#     --gpu 0

python train_semseg.py \
    --model pointnet2_sem_seg \
    --batch_size 16 \
    --npoint 4096 \
    --epoch 128 \
    --learning_rate 0.001 \
    --optimizer Adam \
    --log_dir pointnet2_leaf_seg \
    --gpu 0
    
