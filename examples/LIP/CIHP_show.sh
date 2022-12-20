export ID_FILE="../../data/LIP/CIHP/instance-level_human_parsing/Training/train_id.txt"
export MASK_ROOT="../../data/LIP/CIHP/instance-level_human_parsing/Training/Categories/"
export SAVE_ROOT="/nas/.cache/czk_temp/CIHP/"
export OUTPUT_DIR="./out"

export BACKGROUND_OF_INPAINT="a road near river"

echo "[log] Start generating background"
python generate.py \
  --save_root=$SAVE_ROOT \
  --background="$BACKGROUND_OF_INPAINT"

echo "[log] Start inpaintings"
python inference.py \
  --id_file=$ID_FILE \
  --mask_root=$MASK_ROOT \
  --save_root=$SAVE_ROOT \
  --model_folder=$OUTPUT_DIR \
  --prompt="a photo of CIHP persons in $BACKGROUND_OF_INPAINT"