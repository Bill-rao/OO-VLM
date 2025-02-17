NUM_SHARDS=1
NUM_GPUS=1
BATCH_SIZE=256
BASE_LR=2e-5

checkpoint_path="./best.pyth"

data_record_folder="/root/i/DataRecord/data_record_videomaev2_giant"
work_path="/root/linux/OO-VLM"
cfg_path="/root/linux/OO-VLM/exp/data_record/videomaev2_giant/config.yaml"
frame_path="/root/h/DataSet/sthv2/frames_origin_framerate"
bbox_sample_dir="/root/d/SthV2/"
bbox_anno_dir="/root/d/SthV2/bounding_box"

# Data list
# Notice the match between the frame_path and datalist
datalist_path="./data_list/sthv2_origin_framerate"
txt_template="sthv2_rgb_{}_split_origin_framerate.txt" # origin framerate

export PYTHONPATH=$PYTHONPATH:./slowfast

python tools/run_net_multi_node.py \
  --init_method tcp://localhost:10125 \
  --cfg $cfg_path \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_BBOX_SAMPLE_DIR $bbox_sample_dir \
  DATA.PATH_TO_BBOX_ANNO_DIR $bbox_anno_dir \
  DATA.PATH_TO_DATA_DIR $datalist_path \
  DATA.PATH_PREFIX $frame_path \
  DATA.LABEL_PATH_TEMPLATE $txt_template \
  DATA.IMAGE_TEMPLATE "{}_{:06d}.jpg" \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 2 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 50 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 3. \
  DATA.TEST_CROP_SIZE 224 \
  TRAIN.ENABLE False \
  TEST.NUM_ENSEMBLE_VIEWS 2 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.BATCH_SIZE $BATCH_SIZE \
  TEST.CHECKPOINT_FILE_PATH $checkpoint_path \
  TEST.ADD_SOFTMAX True \
  DATA.MC True \
  RNG_SEED 42 \
  OUTPUT_DIR $work_path \
  TRAIN.IS_RESET_EPOCH False \
  DATARECORDER.OUTPUT_FOLDER_PATH $data_record_folder \
