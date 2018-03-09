COCO:
	wget -c -P data http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip http://msvocds.blob.core.windows.net/coco2014/train2014.zip
	git clone https://github.com/pdollar/coco data/coco
	cd data && unzip instances_train-val2014.zip && unzip instances_minival2014.json.zip && unzip train2014.zip && cd coco/PythonAPI && make
	mkdir -p data/coco/images
	mv data/train2014 data/coco/images
	mv data/annotations data/coco/annotations
