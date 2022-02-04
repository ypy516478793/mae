# val 300, train 100, classes 500
python submitit_finetune.py \
    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr100_cls500 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 100 \
    --partition 'batch' \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 100 --num_val 300 --nb_classes 500

# val 300, train 200, classes 500
python submitit_finetune.py \
    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr200_cls500 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 100 \
    --partition 'batch' \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 200 --num_val 300 --nb_classes 500


# val 300, train 500, classes 500
python submitit_finetune.py \
    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr500_cls500 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 100 \
    --partition 'batch' \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 500 --num_val 300 --nb_classes 500


# val 300, train 1000, classes 500
python submitit_finetune.py \
    --job_dir jobdir/vit_base_patch16_e100_input224_imagenet_tr1000_cls500 \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 100 \
    --partition 'batch' \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path imagenet \
    -d imagenet_limit \
    --num_tr 1000 --num_val 300 --nb_classes 500