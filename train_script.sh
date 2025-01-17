#python submitit_pretrain.py \
#    --resume 'jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
#    --job_dir jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_base_patch16 \
#    --input_size 64 \
#    --norm_pix_loss \
#    --mask_ratio 0.75 \
#    --epochs 800 \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    --partition 'batch' \
#    --dataset image_lesion_combined_datasets \

#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_full \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --partition 'batch' \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval --data_path imagenet


#python submitit_linprobe.py \
#    --job_dir jobdir/vit_base_patch16_lb_e90_input224_imagenet_full \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 1024 \
#    --model vit_base_patch16 --cls_token \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 90 \
#    --partition 'batch' \
#    --blr 0.1 \
#    --weight_decay 0.0 \
#    --dist_eval --data_path imagenet

#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
#    --batch_size 128 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path imagenet

#
## Pretrain_mr5
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr5 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung \
#    --input_size 32 \
#    --mask_ratio 0.5
#
## Pretrain_mr6
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr6 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung \
#    --input_size 32 \
#    --mask_ratio 0.6
#
## Pretrain_mr7
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr7 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung \
#    --input_size 32 \
#    --mask_ratio 0.7
#
## Pretrain_mr8
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr8 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung \
#    --input_size 32 \
#    --mask_ratio 0.8

## Finetuning imagenet_limit
#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr100 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --partition 'batch' \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval --data_path imagenet \
#    -d imagenet_limit \
#    --num_tr 100 --num_val 300 --nb_classes 5

## Finetuning lung -> luna_nodule
#python submitit_finetune.py \
#    --job_dir jobdir/finetune_lung_mr75_lunaV6/vit_large_patch16_e500_input32_luna_blr1e3 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model vit_large_patch16 \
#    --finetune jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung/checkpoint-799.pth \
#    --epochs 500 \
#    --partition 'batch' \
#    --blr 1e-3 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval \
#    -d luna_nodule \
#    --nb_classes 2 \
#    --input_size 32

## Train luna_nodule from scratch
#python submitit_finetune.py \
#    --job_dir jobdir/scratch_lung/vit_large_patch16_e500_input32_luna_blr1e2 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model vit_large_patch16 \
#    --epochs 500 \
#    --partition 'batch' \
#    --blr 1e-2 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval \
#    -d luna_nodule \
#    --nb_classes 2 \
#    --input_size 32

## Train luna_nodule from scratch with smaller path (p4_)
#python submitit_finetune.py \
#    --job_dir jobdir/scratch_lung/vit_large_patch4_e500_input32_luna_blr1e3 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 32 \
#    --model vit_large_patch4 \
#    --epochs 500 \
#    --partition 'batch' \
#    --blr 1e-3 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval \
#    -d luna_nodule \
#    --nb_classes 2 \
#    --input_size 32

## Pretrain_mr75 with smaller path (p4_)
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung/vit_large_patch4_e800_input32_luna_mr75 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 32 \
#    --model mae_vit_large_patch4 \
#    --norm_pix_loss \
#    --mask_ratio 0.75 \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung \
#    --input_size 32 \

## Pretrain_mr75 on lung nodule only
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_lung_nodule/vit_large_patch8_e2400_crop32_lung_mr75_blr1.5e4_wu500 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 128 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 2400 \
#    --partition 'batch' \
#    --warmup_epochs 500 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d lung_nodule \
#    --input_size 32 \
#    --mask_ratio 0.75

# Pretrain_mr75 on lung nodule only
python submitit_pretrain.py \
    --job_dir jobdir/pretrain_lung_nodule/vit_large_patch8_e3600_crop32_lung_mr75_blr1.5e4_wu1000 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --epochs 3600 \
    --partition 'batch' \
    --warmup_epochs 1000 \
    --blr 1.5e-4 --weight_decay 0.05 \
    -d lung_nodule \
    --input_size 32 \
    --mask_ratio 0.75

## Pretrain_mr75 on lung nodule only
CUDA_VISIBLE_DEVICES=0,4,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --batch_size 128 --model mae_vit_large_patch16 \
    --resume jobdir/pretrain_lung_nodule/vit_large_patch8_e3600_crop32_lung_mr75_blr1.5e4_wu1000/checkpoint-3599.pth \
    --epochs 3600 --blr 1.5e-4 --weight_decay 0.05 \
    --output_dir jobdir/pretrain_lung_nodule/vit_large_patch8_e3600_crop32_lung_mr75_blr1.5e4_wu1000_debug \
    -d lung_nodule --input_size 32 --mask_ratio 0.75 --num_workers 0


## Train imagenet_limit from scratch
#python submitit_finetune.py \
#    --job_dir jobdir/scratch_imagenet/vit_base_patch16_e800_input224_imagenet_tr200_cls5_blr1e3 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 64 \
#    --model vit_large_patch16 \
#    --epochs 800 \
#    --partition 'batch' \
#    --blr 1e-3 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval --data_path imagenet \
#    -d imagenet_limit \
#    --num_tr 100 --num_val 300 --nb_classes 5


#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
#    --batch_size 32 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --blr 5e-4 --layer_decay 0.65 \ls

#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path data/ILSVRC2012 \
#    -d imagenet_limit \
#    --num_tr 100 --num_val 300 --nb_classes 5
#
#python main_finetune.py \
#    --batch_size 32 --model vit_base_patch16 --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 \
#    --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path imagenet --device cpu

#python main_finetune.py \
#    --batch_size 32 --model vit_large_patch16 --finetune jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung/checkpoint-799.pth \
#    --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 \
#    --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --device cpu -d luna_nodule --nb_classes 2 --input_size 32


#python main_pretrain_single.py \
#    --batch_size 1 --model mae_vit_large_patch16 --resume mae_visualize_vit_large.pth \
#    --epochs 100 --blr 1.5e-4 --weight_decay 0.05 \
#    --output_dir jobdir/pretrain_imagenet_single/vit_large_patch16_e800_ft100 \
#    --data_path imagenet -d imagenet_limit --num_tr 1 --num_val 1 --nb_classes 1