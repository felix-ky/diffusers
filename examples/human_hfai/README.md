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

## TODOs

- [ ] implement inference
- [ ] tune lr schedules and optimizer hyperparams
- [ ] fix all TODOs left in the code
- [ ] use whole the COCO Densepose dataset (by generating samples OTF)
- [ ] instance-aware random crop
- [ ] add uv coordinate conditions with densepose
- [ ] prompt tuning/adapter
- [ ] example prompt/text prompt engineering
- [ ] 3D conditions like apple neuman-ml

