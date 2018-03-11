export MPLBACKEND=Agg

# multiscale
#python trainval_net.py --cuda --anno ./data/coco/annotations/instances_train2014_5000.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0 --epochs 11 --s 2 --multiscale

# no multiscale
python trainval_net.py --cuda --anno ./data/coco/annotations/instances_train2014_5000.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0 --epochs 11 --s 2
