export WORK_DIR=$(pwd)
export DATASET_NAME=bdd100k
export TASK=ins_seg

export IMG_DIR=${WORK_DIR}/datasets/${DATASET_NAME}/images/10k
export ANNOT_DIR=${WORK_DIR}/datasets/${DATASET_NAME}/labels_coco/ins_seg
export CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_SAM_3x.py

python3 evaluate.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --model_weight /home/giakhang/Downloads/model_best.pth 