export MPLBACKEND=Agg

python trainval_net.py --cuda --anno ./data/coco/annotations/instances_trainval2014_mini.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0
