```
git clone https://github.com/Eikewi/slimmable-ConvNeXt.git
cd slimmable-ConvNeXt
```

```
conda create -n convnext python=3.8.20
conda activate convnext
pip install -r requirements.txt
```

Basemodel training:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --model convnext_tiny --drop_path 0.1 --batch_size 256 --epochs 150  --nb_classes 1000  --lr 4e-3 --update_freq 4 --model_ema false --model_ema_eval false --enable_wandb true  --data_path /data22/datasets/ilsvrc2012/ --dist_url tcp://127.0.0.1:29507
```

Finales Training:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --model convnext_tiny --drop_path 0.1 --batch_size 256 --epochs 150 --nb_classes 1000 --lr 4e-3 --update_freq 4 --model_ema false --model_ema_eval false --enable_wandb true --finetune ./base_model.pth  --data_path /data22/datasets/ilsvrc2012/ --dist_url tcp://127.0.0.1:29507
```