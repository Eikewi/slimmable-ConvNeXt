```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 main.py   --model convnext_tiny --drop_path 0.1 --batch_size 64 --epochs 150  --nb_classes 1000  --lr 4e-3 --update_freq 4 --model_ema false --model_ema_eval false --enable_wandb true --eval false --find_slim false   --data_path /data22/datasets/ilsvrc2012/ --dist_url tcp://127.0.0.1:29507


```
