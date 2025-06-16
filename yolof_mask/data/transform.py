import numpy as np
from fvcore.transforms.transform import (
    NoOpTransform,
    Transform,
)
import detectron2.data.transforms as T 

class AlbumentationsTransform(Transform):
    """
    A class that wraps an albumentations transform.
    """

    def __init__(self, aug, params, img_height, img_width):
        """
        Args:
            aug (albumentations.BasicTransform):
            params (dict): parameters for the albumentations transform
            img_height (int): height of the image to be transformed
            img_width (int): width of the image to be transformed
        """
        self.aug = aug
        self.params = params
        self.img_height_orig = img_height
        self.img_width_orig = img_width
        self.img_height_tfm = None
        self.img_width_tfm = None

    def apply_image(self, image):
        res = self.aug.apply(image, **self.params)
        self.img_height_tfm, self.img_width_tfm = res.shape[:2]
        return res

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        assert (
            self.img_height_tfm is not None and self.img_width_tfm is not None
        ), "Image must be transformed first before applying bounding box transformations."

        # Note that just like the base Transform class, bbox coordinates are
        # expected to be in XYXY_ABS format. Albumentations expects XYXY_REL format.

        h, w = self.img_height_orig, self.img_width_orig
        box_rel = box / [w, h, w, h]

        try:
            res = np.array(self.aug.apply_to_bboxes(box_rel.tolist(), **self.params))
        except AttributeError:
            return box

        # In case the albumentations transform drops the bounding box (e.g. because it is out of bounds),
        # we need to ensure that the number of boxes is the same as before the transform.
        if len(res) == 0:
            res = np.zeros((1, 4))  # idea is that this box will get filtered out later

        h, w = self.img_height_tfm, self.img_width_tfm
        res = res * [w, h, w, h]

        return res

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    
class AlbumentationsWrapper(T.Augmentation):
    """
    Wrap an augmentation from the albumentations library:
    https://github.com/albumentations-team/albumentations/.
    Image, Bounding Box and Segmentation are supported.
    """

    def __init__(self, aug, **kwargs):
        """
        Args:
            aug (albumentations.BasicTransform): albumentations transform
        """
        self._aug = aug
        self.kwargs = kwargs

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = self.prepare_params(image)
            h, w = image.shape[:2]
            return AlbumentationsTransform(self._aug, params, h, w)
        else:
            return NoOpTransform()

    def prepare_params(self, image):
        params = self._aug.get_params()
        targets_as_params = {"image": image}
        params_dependent_on_targets = self._aug.get_params_dependent_on_data(
            params=targets_as_params,
            data=targets_as_params
        )
        params.update(params_dependent_on_targets)
        params = self._aug.update_transform_params(params, {"image": image})
        params.update(**self.kwargs)
        return params
