import logging
import itertools
import torch.utils.data as torchdata

from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.build import filter_images_with_only_crowd_annotations, filter_images_with_few_keypoints
from detectron2.data import ( 
    DatasetCatalog, 
    MetadataCatalog, 
    load_proposals_into_dataset,
    print_instances_class_histogram
)

def get_filtered_detection_dataset_dicts(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
    **kwargs
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    available_datasets = DatasetCatalog.keys()
    names_set = set(names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning(
            "The following dataset names are not registered in the DatasetCatalog: "
            f"{names_set - available_datasets}. "
            f"Available datasets are {available_datasets}"
        )

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    min_box_area = kwargs.get("min_box_area", 0)

    num_filtered = 0
    new_dicts = []
    for dataset_dict in dataset_dicts:
        new_dict = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            new_annos = []
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                box_xywh = anno.get("bbox", None)
                if box_xywh is None:
                    continue
                box_area = box_xywh[2] * box_xywh[3]
                if box_area > min_box_area:
                    new_annos.append(anno)
                else:
                    num_filtered += 1
            if len(new_annos) == 0:
                continue
            new_dict["annotations"] = new_annos

        new_dicts.append(new_dict)
    if num_filtered > 0:
        logger = logging.getLogger(__name__)
        logger.warning(f"filtered out {num_filtered} instances with box_area <= {min_box_area}")

    dataset_dicts = new_dicts

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts
