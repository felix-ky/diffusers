export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR1="imgs/dog"
export INSTANCE_DIR2="imgs/cat"
export OUTPUT_DIR="outputs/multix2"

accelerate launch train_dreambooth_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir1=$INSTANCE_DIR1 \
  --instance_data_dir2=$INSTANCE_DIR2 \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt1="a photo of sks dog" \
  --instance_prompt2="a photo of yhb cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800
