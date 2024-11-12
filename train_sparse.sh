CUDA_VISIBLE_DEVICES=7 python scripts/cifar10/qmobilevit/train_moe.py

# export CUDA_VISIBLE_DEVICES=7; 
# python train.py configs/cifar10/qmobilevit/train/moe.yml \
#     --optimizer.lr=0.002 \
#     --run.random_state=42 \
#     --model.conv_cfg.w_bit=8 \
#     --model.linear_cfg.w_bit=8 \
#     --model.matmul_cfg.w_bit=8 \
#     --model.conv_cfg.out_bit=8 \
#     --model.linear_cfg.out_bit=8 \
#     --model.matmul_cfg.out_bit=8 \
#     --model.conv_cfg.in_bit=8 \
#     --model.linear_cfg.in_bit=8 \
#     --model.matmul_cfg.in_bit=8 \
#     --checkpoint.model_comment=lr-0.0020_wb-8_run-1