export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_ID_FILE="../../data/LIP/LIP/TrainVal_images/train_id.txt"
export INSTANCE_DIR="../../data/LIP/LIP/TrainVal_images/train_images/"
export INSTANCE_MASK_DIR="../../data/LIP/LIP/TrainVal_parsing_annotations/train_segmentations/"
export SAVE_ROOT="/nas/.cache/czk_temp/LIP/"
export OUTPUT_DIR="./out_3000"

export BACKGROUND_OF_INPAINT="a road near river"

export GPU_IDS="0,1,2,3"

echo "[log] Start training"
accelerate launch \
  --config_file="./config.yaml" \
  --gpu_ids=$GPU_IDS \
  training.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_id_file=$INSTANCE_ID_FILE \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_data_mask_dir=$INSTANCE_MASK_DIR \
  --save_root=$SAVE_ROOT \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a LIP person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000

echo "[log] Start generating background"
python generate.py \
  --save_root=$SAVE_ROOT \
  --background="$BACKGROUND_OF_INPAINT"

echo "[log] Start inpaintings"
python inference.py \
  --id_file=$INSTANCE_ID_FILE \
  --mask_root=$INSTANCE_MASK_DIR \
  --save_root=$SAVE_ROOT \
  --model_folder=$OUTPUT_DIR \
  --prompt="a photo of a LIP person in $BACKGROUND_OF_INPAINT"