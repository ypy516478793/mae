## val 300, train 100, classes 500
#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr100_cls500 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 32 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --partition 'batch' \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval --data_path imagenet \
#    -d imagenet_limit \
#    --num_tr 100 --num_val 300 --nb_classes 500
#
## val 300, train 200, classes 500
#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr200_cls500 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 64 \
#    --model vit_base_patch16 \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 100 \
#    --partition 'batch' \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --dist_eval --data_path imagenet \
#    -d imagenet_limit \
#    --num_tr 200 --num_val 300 --nb_classes 500
#
#
## val 300, train 500, classes 500
#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr500_cls500 \
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
#    --num_tr 500 --num_val 300 --nb_classes 500
#
#
## val 300, train 1000, classes 500
#python submitit_finetune.py \
#    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr1000_cls500 \
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
#    --num_tr 1000 --num_val 300 --nb_classes 500



## Pretrain_mr75 on lung nodule only
#python submitit_pretrain.py \
#    --job_dir jobdir/pretrain_imagenet_tr100_val20_cls5/vit_large_patch8_e800_mr75 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 125 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --epochs 800 \
#    --partition 'batch' \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    -d imagenet_limit \
#    --data_path imagenet \
#    --num_tr 100 --num_val 20 --nb_classes 5 \
#    --mask_ratio 0.75 \
#    --resume mae_visualize_vit_large.pth

### linprobe blr 0.1, train 100, classes 5
#python submitit_linprobe.py \
#    --job_dir jobdir/linprobe_imagenet/vit_base_patch16_lb_e90_input224_imagenet_tr100_cls5 \
#    --ngpus 4 \
#    --nodes 1 \
#    --timeout 17280 \
#    --batch_size 64 \
#    --model vit_base_patch16 --cls_token \
#    --finetune mae_pretrain_vit_base.pth \
#    --epochs 90 \
#    --partition 'batch' \
#    --blr 0.1 \
#    --weight_decay 0.0 \
#    --dist_eval --data_path imagenet \
#    -d imagenet_limit \
#    --num_tr 100 --num_val 300 --nb_classes 5

## linprobe lr6.4, train 100, classes 5
python submitit_linprobe.py \
    --job_dir jobdir/linprobe_imagenet/vit_large_patch16_tr100_val20_ft50_lr6.4_meta \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 125 \
    --model vit_large_patch16 --cls_token \
    --finetune jobdir/pretrain_imagenet_tr100_val20_cls5/vit_large_patch8_e800_mr75/checkpoint-799.pth \
    --epochs 50 \
    --partition 'batch' \
    --lr 6.4 \
    --weight_decay 0.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 100 --num_val 20 --nb_classes 5


## linprobe lr6.4, train 100, classes 5
#python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
#    --batch_size 125 --model vit_large_patch16 --cls_token --finetune mae_visualize_vit_large.pth \
#    --epochs 50 --lr 6.4 --weight_decay 0.0 --dist_eval \
#    --output_dir jobdir/linprobe_imagenet/vit_large_patch16_tr100_val20_ft50_lr6.4 \
#    --data_path imagenet -d imagenet_limit --num_tr 100 --num_val 20 --nb_classes 5 --num_workers 0

### finetune ssl on single lung
#python -m torch.distributed.launch --nproc_per_node=1 main_pretrain_single.py \
#    --batch_size 1 --model mae_vit_large_patch16 \
#    --resume jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung/checkpoint-799.pth \
#    --epochs 100 --blr 5e-2 --weight_decay 0.05 \
#    --output_dir jobdir/pretrain_lung_single/vit_large_patch16_e800_ft100_blr5e2_wu40 \
#    -d lung --num_tr 1 --num_val 1 --warmup_epochs 40 --input_size 32
#
#python -m torch.distributed.launch --nproc_per_node=1 main_pretrain_meta.py \
#    --batch_size 4 --model mae_vit_large_patch16 \
#    --finetune mae_visualize_vit_large.pth --epochs 100 \
#    --blr_ssl 5e-2 --lr 1 --weight_decay 0.05 \
#    --output_dir jobdir/meta_imagenet/vit_large_patch16_e800_ft100_blr5e2_lr1_wu40 \
#    --data_path imagenet -d imagenet_limit --num_tr 100 --num_val 300 --nb_classes 5 \
#    --warmup_epochs 40 --num_workers 0
#
#python -m torch.distributed.launch --nproc_per_node=2 main_pretrain_meta.py \
#    --batch_size 16 --model mae_vit_large_patch16 \
#    --finetune mae_visualize_vit_large.pth --epochs 100 --epochs_ssl 0 \
#    --blr_ssl 5e-2 --lr 1 --weight_decay 0.05 \
#    --output_dir jobdir/meta_imagenet/vit_large_patch16_tr100_val20_bft0_mft100_bblr5e2_mlr1_bwu40_mwu10 \
#    --data_path imagenet -d imagenet_limit --num_tr 100 --num_val 20 --nb_classes 5 \
#    --warmup_epochs 10 --warmup_epochs_ssl 40 --num_workers 10


### linprobe
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
#   --batch_size 125 --model vit_large_patch16 --cls_token \
#   --finetune mae_pretrain_vit_large.pth --epochs 50 \
#   --lr 6.4 --weight_decay 0.0 --dist_eval \
#   --output_dir jobdir/linprobe_imagenet/vit_large_patch16_tr100_val20_ft50_lr6.4_noAug \
#   --data_path imagenet -d imagenet_limit --num_tr 100 --num_val 20 --nb_classes 5

### extract features and save embedding
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_features.py \
#    --batch_size 125 --model vit_large_patch16 --cls_token \
#    --finetune mae_pretrain_vit_large.pth --epochs 50 \
#    --lr 6.4 --weight_decay 0.0 --dist_eval --eval \
#    --output_dir jobdir/features_imagenet/vit_large_patch16_tr100_val20_ft50_lr6.4 \
#    --data_path imagenet -d imagenet_limit_with_name --num_tr 100 --num_val 20 --nb_classes 5 --num_workers 0

### linprobe use features directly
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_features.py \
#    --batch_size 125 --model vit_large_patch16_use_features --cls_token \
#    --finetune mae_pretrain_vit_large.pth --epochs 100 \
#    --lr 10 --weight_decay 0.0 --dist_eval \
#    --output_dir jobdir/features_imagenet/vit_large_patch16_tr100_val20_ft100_lr10_metab1_wu40 \
#    --data_path imagenet_features/meta_b1 -d imagenet_limit_use_features \
#    --num_tr 100 --num_val 20 --nb_classes 5 --num_workers 0 --warmup_epochs 40


### meta save ssl features
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_meta_features.py \
#    --batch_size 1 --model mae_vit_large_patch16 --finetune mae_visualize_vit_large.pth \
#    --epochs 1 --epochs_ssl 100 --blr_ssl 5e-2 --lr 1 --weight_decay 0.05 \
#    --output_dir jobdir/meta_imagenet/vit_large_patch16_b1_tr100_val20_bft100_mft1_bblr5e2_mlr1_bwu40_mwu10 \
#    --data_path imagenet -d imagenet_limit_with_name --num_tr 100 --num_val 20 --nb_classes 5 \
#    --warmup_epochs 10 --warmup_epochs_ssl 40 --num_workers 10 --feature_dir imagenet_features/meta_b1
