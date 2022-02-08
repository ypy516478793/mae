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

## linprobe blr 10, train 100, classes 5
python submitit_linprobe.py \
    --job_dir jobdir/linprobe_imagenet/vit_base_patch16_lb_e90_input224_imagenet_tr100_cls5_blr10 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 64 \
    --model vit_base_patch16 --cls_token \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 90 \
    --partition 'batch' \
    --blr 10 \
    --weight_decay 0.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 100 --num_val 300 --nb_classes 5

## finetune ssl on single lung
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain_single.py \
    --batch_size 1 --model mae_vit_large_patch16 \
    --resume jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung/checkpoint-799.pth \
    --epochs 100 --blr 5e-2 --weight_decay 0.05 \
    --output_dir jobdir/pretrain_lung_single/vit_large_patch16_e800_ft100_blr5e2_wu40 \
    -d lung --num_tr 1 --num_val 1 --warmup_epochs 40 --input_size 32