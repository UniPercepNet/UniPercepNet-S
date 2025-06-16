import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances, pairwise_iou, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.dense_detector import permute_to_N_HWA_K  # noqa
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.matcher import Matcher
from detectron2.checkpoint import DetectionCheckpointer

from yolof_mask.modeling.backbone import ResNet, VoVNet
from .yolof_mask import YOLOF_Mask, select_foreground_proposals

class Panoptic_YOLOF_Mask(YOLOF_Mask):
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        combine_overlap_thresh: float = 0.5,
        combine_stuff_area_thresh: float = 4096,
        combine_instances_score_thresh: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold

        Other arguments are the same as :class:`GeneralizedRCNN`.
        """
        super().__init__(**kwargs)
        self.sem_seg_head = sem_seg_head
        # options when combining instance & semantic outputs
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """

        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)

        if isinstance(self.backbone, ResNet):
            features = features['res5']
        else:
            print("Invalid type of backbone")
            return

        features_p5 = self.encoder(features)

        assert "sem_seg" in batched_inputs[0], "Semantic segmenations are missing in training!"
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg,
            self.backbone.size_divisibility,
            self.sem_seg_head.ignore_value,
            self.backbone.padding_constraints,
        ).tensor

        _, sem_seg_losses = self.sem_seg_head(features_p5, gt_sem_seg)

        assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        box_cls, box_delta = self.decoder(features_p5)
        anchors = self.anchor_generator([features_p5])

        # Proposals
        pred_logits = [permute_to_N_HWA_K(box_cls, self.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(box_delta, 4)]

        indices = self.get_ground_truth(anchors, pred_anchor_deltas, gt_instances)
        proposal_losses = self.losses(
            indices, gt_instances, anchors, pred_logits, pred_anchor_deltas
        )

        # Mask
        proposals = super().inference([box_cls], [box_delta], anchors, images.image_sizes)
        proposals = self.label_and_sample_proposals(proposals, gt_instances) 

        # TODO: Need to change this logit, we dont need negative proposals so 
        # we can remove all of them before using ROI Align
        proposal_boxes = [x.pred_boxes for x in proposals]
        box_features = self.pooler([features_p5], proposal_boxes)
        del features_p5
        del features
        proposals, fg_selection_masks = select_foreground_proposals(
            proposals, self.num_classes
        )
        # The mask loss is only defined on foreground proposals, 
        # so we need to select out the foreground features.
        mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
        del box_features
        ins_mask_loss = self.mask_head(mask_features, proposals)

        losses = sem_seg_losses
        losses.update(proposal_losses)
        losses.update(ins_mask_loss)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if isinstance(self.backbone, ResNet):
            features = features['res5']
        else:
            print("Invalid type of backbone")
            return

        features_p5 = self.encoder(features)
        box_cls, box_delta = self.decoder(features_p5)
        anchors = self.anchor_generator([features_p5])

        sem_seg_results, _ = self.sem_seg_head(features_p5, None)
        proposals = super().inference([box_cls], [box_delta], anchors, images.image_sizes)
        proposal_boxes = [x.pred_boxes for x in proposals]
        box_features = self.pooler([features_p5], proposal_boxes)
        detector_results = self.mask_head(box_features, proposals)

        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results