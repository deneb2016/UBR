export MPLBACKEND=Agg

#no_set1
#python trainval_net.py --cuda --anno ./data/coco/annotations/instances_train2014_set1_5000.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0 --epochs 11 --s 1 --multiscale

# no_set2
#python trainval_net.py --cuda --anno ./data/coco/annotations/instances_train2014_set2_5000.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0 --epochs 11 --s 2 --multiscale

# no_set3
python trainval_net.py --cuda --anno ./data/coco/annotations/instances_train2014_set3_5000.json --save_dir ./data/checkpoints --gen_box_var 0.35 --iou_th 0.4 --hard_ratio 0 --epochs 11 --s 3 --multiscale

