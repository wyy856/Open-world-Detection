复现：186

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

cd /volume1/wyy/prob

conda activate prob

CUDA_VISIBLE_DEVICES=0 nohup python -u main_open_world.py --output_dir exps/MOWODB/PROB/t1_official --dataset TOWOD --data_root ./data/OWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --num_workers 4 --batch_size 2 --nc_epoch 10 > train_t1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_open_world.py --output_dir exps/MOWODB/PROB/t1_official --dataset TOWOD --data_root ./data/OWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --num_workers 4 --batch_size 1 --nc_epoch 10 > train_t1.log 2>&1 &


tail -f train_t1.log

CUDA_VISIBLE_DEVICES=1 nohup python -u main_open_world.py     --output_dir exps/MOWODB/PROB/t1_B4     --dataset TOWOD     --data_root ./data/OWOD     --PREV_INTRODUCED_CLS 0     --CUR_INTRODUCED_CLS 20     --train_set owod_train_matched     --test_set owod_test_matched     --epochs 41     --model_type prob     --obj_loss_coef 8e-4     --obj_temp 1.3     --num_workers 4     --batch_size 4     --lr 8e-5     --lr_backbone 1.6e-5 > train_B4.log 2>&1 & 



测试：
export CUDA_VISIBLE_DEVICES=0 && python -u main_open_world.py \
--output_dir "exps/MOWODB/PROB/eval_results_T1" \
--dataset TOWOD \
--PREV_INTRODUCED_CLS 20 \
--CUR_INTRODUCED_CLS 20 \
--train_set "owod_t1_train" \
--test_set "owod_all_task_test" \
--model_type "prob" \
--obj_loss_coef 8e-4 \
--obj_temp 1.3 \
--pretrain "exps/MOWODB/PROB/t1_official/checkpoint0040.pth" \
--eval \
--batch_size 2 \
--num_queries 100

