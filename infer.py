import argparse
import cv2
import os

from detectron2.config import LazyConfig
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from yolof_mask.engine.default_predictor import DefaultPredictor
from yolof_mask.utils.visualizer import Visualizer, ColorMode

def main(**kwargs):
    config_path = kwargs.get("config")
    device = kwargs.get("device")
    input_path = kwargs.get("input_path")
    output_path = kwargs.get("output_path")
    model_weight = kwargs.get("model_weight")
    score_threshold = kwargs.get("score_threshold")
    max_dets = kwargs.get("max_dets")
    vis_result = kwargs.get("vis_result")
    task = kwargs.get("task")
    dataset_name = kwargs.get("dataset_name")

    cfg = LazyConfig.load(config_path)
    if device:
        cfg.train.device = f"cuda:{device}"
    else:
        cfg.train.device = "cpu"
    
    assert os.path.exists(model_weight), f"`{model_weight}` does not exist!!!"
    assert os.path.exists(input_path), f"`{input_path}` does not exist!!!"

    assert dataset_name in ["bdd100k"], "Currently, we just support `bdd100k` dataset only!"
    assert task in ["ins_seg"], "Currently, we just support Instance Segmentation task"

    if dataset_name == "bdd100k":
        class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    cfg.train.init_checkpoint = model_weight
    if score_threshold:
        cfg.model.score_thresh_test = score_threshold
    if max_dets:
        cfg.model.max_detections_per_image = max_dets
    predictor = DefaultPredictor(cfg) 

    img = cv2.imread(input_path)
    predictions = predictor(img)
    v = Visualizer(img, class_names=class_names)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    img_result = out.get_image()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, img_result)

    if vis_result:
        cv2.imshow("Prediction result", img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # TODO: Support `video` and `webcam`
    parser = argparse.ArgumentParser(description="Infer an image using a configuration file.")
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=None,
        help="Device to use. None mean `cpu`."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Image input path."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=None,
        help="Ouput path to save."
    )
    parser.add_argument(
        "--model_weight",
        type=str,
        required=True,
        help="Model weight path."
    )
    parser.add_argument(
        '--score_threshold', 
        type=float, 
        default=None, 
        help='Detection threshold.'
    )
    parser.add_argument(
        '--max_dets', 
        type=int, 
        default=None, 
        help='Max detections.'
    )
    parser.add_argument(
        '--vis_result', 
        action="store_true",
        help='Visualize prediction.'
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bdd100k",
        required=True,
        help="Name of data to train."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ins_seg",
        required=True,
        help="Task to train."
    )

    args = parser.parse_args()
    main(**vars(args))