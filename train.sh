CUDA_VISIBLE_DEVICES=3 \
python step2_train_4DGSFFMotion.py\
 --config configs/ff4dgsmotion.yaml\
 --output_dir results/FF4DGSMotion/train \
 --resume results/FF4DGSMotion/train/20251216_020958/best.pth