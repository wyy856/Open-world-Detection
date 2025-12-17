复现：186

cd /volume1/wyy/OWOBJ

conda activate owobj

T1阶段：

nohup bash -c 'CUDA_VISIBLE_DEVICES=1 GPUS_PER_NODE=2 MASTER_PORT=29502 ./tools/run_dist_launch.sh 1 configs/M_OWOD_BENCHMARK.sh' > training_owobj_t1.log 2>&1 &

tail -f training_owobj_t1.log

/volume1/wyy/OWOBJ/configs/M_OWOD_BENCHMARK.sh文件：

#!/usr/bin/env bash

echo running training of OWOBJ, M-OWODB dataset

set -x

EXP_DIR=exps/MOWODB/OWOBJ
PY_ARGS=${@:1}
WANDB_NAME=PROB_t1

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" \
    --dataset OWDETR \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 19 \
    --data_root 'data/OWOD' \
    --train_set 'owod_t1_train' \
    --test_set 'test' \
    --epochs 41 \
    --cls_loss_coef 2 \
    --focal_alpha 0.25 \
    --model_type 'sketch' \
    --obj_loss_coef 8e-4 \
    --obj_temp 1.3 \
    --obj_kl_div 0.1 \
    --exemplar_replay_max_length 850 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_cur_file "learned_owod_t1_ft.txt" \
    --lr 0.0001 \
    --weight_decay 1e-4 \
    --lr_drop 17 \
    --clip_max_norm 5.0 \
    --start_epoch 1 \
    --override_resumed_lr_drop True \
    --resume "${EXP_DIR}/t1/checkpoint.pth" \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t1_resume.txt

生成t1的learned_owod_t1_ft.txt文件

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python -u main_open_world.py \
    --output_dir \"exps/MOWODB/OWOBJ/t1\" \
    --dataset OWDETR \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 19 \
    --data_root 'data/OWOD' \
    --train_set 'owod_t1_train' \
    --test_set 'test' \
    --epochs 1 \
    --model_type 'sketch' \
    --exemplar_replay_max_length 850 \
    --exemplar_replay_dir \"PROB_t1\" \
    --exemplar_replay_cur_file \"learned_owod_t1_ft.txt\" \
    --resume \"exps/MOWODB/OWOBJ/t1/checkpoint.pth\" \
    --exemplar_replay_selection \
    2>&1 | tee \"exps/MOWODB/OWOBJ/logs/generate_t1_exemplar_final_write.txt\"" &
    
tail -f exps/MOWODB/OWOBJ/logs/generate_t1_exemplar_final_write.txt

修改！！！

t1训练命令:

mkdir -p exps/MOWODB/OWOBJ/logs/ && nohup python -u main_open_world.py --output_dir "exps/MOWODB/OWOBJ/t1_rerun" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root 'data/OWOD' --train_set 'owod_t1_train' --test_set 'test' --epochs 41 --cls_loss_coef 2 --focal_alpha 0.25 --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_t1 --exemplar_replay_cur_file "learned_owod_t1_ft.txt" --lr 0.0001 --weight_decay 1e-4 --lr_drop 17 --clip_max_norm 5.0 --resume "models/dino_resnet50_pretrain.pth" 2>&1 | tee exps/MOWODB/OWOBJ/logs/t1_static_rerun.txt > /dev/null &
 
tail -f exps/MOWODB/OWOBJ/logs/t1_static_rerun.txt


T2阶段：

训练：

nohup bash -c "CUDA_VISIBLE_DEVICES=2 python -u main_open_world.py \
    --output_dir \"exps/MOWODB/OWOBJ/t2\" \
    --dataset OWDETR \
    --PREV_INTRODUCED_CLS 19 \
    --CUR_INTRODUCED_CLS 20 \
    --data_root 'data/OWOD' \
    --train_set 'owod_t2_train' \
    --test_set 'test' \
    --epochs 50 \
    --model_type 'sketch' \
    --exemplar_replay_max_length 850 \
    --exemplar_replay_dir \"PROB_t1\" \
    --exemplar_replay_prev_file \"learned_owod_t1_ft.txt\" \
    --resume \"exps/MOWODB/OWOBJ/t1/checkpoint.pth\" \
    2>&1 | tee \"exps/MOWODB/OWOBJ/logs/t2_train_GPU2_rerun.txt\"" &

tail -f exps/MOWODB/OWOBJ/logs/t2_train_GPU2_rerun.txt

t2训练错误版本，未添加learn文件：

CUDA_VISIBLE_DEVICES=1 nohup bash configs/M_OWOD_BENCHMARK.sh > training_t2.log 2>&1 &

tail -f  training_t2.log 2

CUDA_VISIBLE_DEVICES=1 python -u main_open_world.py --output_dir "exps/MOWODB/OWOBJ/t1" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root 'data/OWOD' --train_set 'owod_t1_train' --test_set 'test' --epochs 1 --model_type 'sketch' --exemplar_replay_max_length 850 --exemplar_replay_dir "PROB_t1" --exemplar_replay_cur_file "learned_owod_t1_ft.txt" --resume "exps/MOWODB/OWOBJ/t1/checkpoint.pth" --eval --exemplar_replay_selection 2>&1 | tee "exps/MOWODB/OWOBJ/logs/generate_t1_exemplar_GPU2_fixed.txt"


