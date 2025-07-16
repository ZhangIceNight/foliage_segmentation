# export CUDA_VISIBLE_DEVICES=1


log_dir="log/sem_seg_mix/"

# pt_hmamba
python test_mix.py \
    --model pointnet2_sem_seg \
    --batch_size 1 \
    --npoint 4096 \
    --log_dir $log_dir \
    --gpu 0 \
    --ckpts "$log_dir/checkpoints/best_model.pth" 

