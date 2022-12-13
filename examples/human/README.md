## Prepare DensePose dataset

First download the [CSE based annotations](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_DATASETS.md#continuous-surface-embeddings-annotations) from detectron2 repo.

Add soft link to local folder
 
```
ln -s $PATH_TO_DENSEPOSE data
ln -s $PATH_TO_COCO data
```

Then convert selected coco subsets into diffuser inpaint dataset by running

```
python prepare_densepose.py
```

Then run the training script
```
./train.sh
```