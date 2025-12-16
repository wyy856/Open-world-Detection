复现：188

cd /icislab/volume2/wyy/OW-DETR

conda activate owdetr

CUDA_VISIBLE_DEVICES=4 nohup python main_open_world.py   --dataset owod   --data_root data/VOC2007/OWOD/   --train_set t1_train   --test_set test   --batch_size 2   --epochs 50   --num_classes 21   --output_dir exps/owod_task1   --num_workers 4   --device cuda   --invalid_cls_logits   --NC_branch   --nc_loss_coef 1.0   --lr 1e-4   --weight_decay 1e-4   --lr_drop 40 > exps/owod_task1/train_gpu450.log 2>&1 &


