import numpy as np
import os, json, cv2, random
import json
import cv2
import os
from tqdm import tqdm
import torch
import torchvision
import mmdet
import mmcv
import mmengine

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from pycocotools.mask import encode as cvt_mask_to_rle
from pycocotools.mask import decode as cvt_rle_to_mask

config_file = './configs/unipercepnet_v2.py'
ckpt_file = '/home/alan_khang/Desktop/unipercepnet_v2/epoch_12.pth'

model = init_detector(config_file, checkpoint=ckpt_file, device='cuda:0')

import numpy as np
import os, json, cv2, random
import json
import cv2
import os
from tqdm import tqdm

from pycocotools.mask import encode as cvt_mask_to_rle
from pycocotools.mask import decode as cvt_rle_to_mask

annot_path = "./datasets/coco2017/coco_test_annotations/image_info_test-dev2017.json"
img_dir = "./datasets/coco2017/coco_test_dev2017/coco_test2017"
annot_info_file = annot_path

results_file = './output/coco_test_infer/detections_test-dev2017_regnetx_tood_1x_results.json'

if os.path.exists(results_file):
    os.remove(results_file)
else:
    fp = open(results_file, 'w')
    fp.close()

with open(annot_info_file, 'r') as file:
    annots = json.load(file)

num_imgs = len(annots['images'])

annType = ['segm','bbox','keypoints']
annType = annType[0]

coco_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def convert_bbox_xyxy_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

with open(results_file, 'w') as file:
    for i, image in enumerate(tqdm(annots['images'])):

        if i == 0:
            file.write('[')
        else:
            file.write(',')

        file_name = image['file_name']
        image_id = image['id']
        results_each_img = []

        if (file_name is None) or (image_id is None):
            print('file_name or image_id is null')

        image_path = os.path.join(img_dir, file_name)
        img = mmcv.imread(image_path, channel_order='bgr')
        predictions = inference_detector(model, img)

        output = predictions.pred_instances
        pred_boxes = output.bboxes.to('cpu')
        pred_masks = output.masks.to('cpu')
        pred_cls_ids = output.labels.to('cpu')
        pred_confs = output.scores.to('cpu')
        assert pred_boxes.size()[0] == pred_masks.size()[0] == pred_cls_ids.size()[0] == pred_confs.size()[0]

        for j in range(pred_boxes.size()[0]):
            box, mask, model_cls_id, conf = pred_boxes[j], pred_masks[j], pred_cls_ids[j], pred_confs[j]
            pred_coco_id = coco_ids[model_cls_id.item()]
            result = {"image_id": image_id,
                      "category_id": pred_coco_id,
                      "score": round(conf.item(), 3)}

            if annType == "bbox":
                result["bbox"] = convert_bbox_xyxy_to_xywh(*box.tolist())
            elif annType == 'segm':
                mask = mask.numpy().astype(np.uint8)
                mask = np.asfortranarray(mask)
                rle = cvt_mask_to_rle(mask)
                rle["counts"] = rle["counts"].decode()
                result["segmentation"] = rle
            else:
                print("Wronge annType")

            results_each_img.append(result)

        # Write result to file
        file.write(json.dumps(results_each_img)[1:-1])

        if i == num_imgs - 1:
            file.write(']')
