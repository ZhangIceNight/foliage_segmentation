# export CUDA_VISIBLE_DEVICES=1

# pt_hmamba
python train_mix.py \
    --model pt_hmamba \
    --batch_size 16 \
    --npoint 4096 \
    --epoch 300 \
    --learning_rate 0.0002 \
    --optimizer AdamW \
    --log_dir "pt_hmamba_$1" \
    --gpu 0 \
    --ckpts 'pretrain.pth'