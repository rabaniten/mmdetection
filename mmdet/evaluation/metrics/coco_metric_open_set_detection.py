import os
import json
import hashlib
import warnings
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
import torch

@METRICS.register_module()
class OpenSetCOCOMetric(CocoMetric):
    """Custom COCO Evaluator for Open-Set Detection Models in MMDetection.

    This class extends CocoMetric to include additional processing for
    mapping text-based labels to COCO categories before performing evaluation.
    """

    def __init__(self, ann_file, outfile_prefix=None, **kwargs):
        """
        Args:
            ann_file (str): Path to the COCO annotation file (GT data).
            outfile_prefix (str, optional): Prefix for saving results.
        """
        super().__init__(ann_file=ann_file, outfile_prefix=outfile_prefix, **kwargs)

        self.ann_file = ann_file 
        
        # Load COCO ground truth annotations
        self.coco_gt = COCO(self.ann_file)


        # mmdetection internally uses labels corresponding to position in `classes` 
        # in config (which should correspond to categories in annotations), 
        # hence, we should not use original category IDs.
        
        # Mapping: category name → category ID
        self.global_prompt_to_index = {
            cat["name"]: cat["id"] for cat in self.coco_gt.dataset["categories"]
        }

        # # Mapping: filename → image ID
        # self.img_ids_dict = {
        #     image["file_name"]: image["id"] for image in self.coco_gt.dataset["images"]
        # }
    
        # # Mapping: category ID → category name
        # self.cat_id_to_name = {
        #     cat["id"]: cat["name"] for cat in self.coco_gt.dataset["categories"]
        # }

        # # Store all possible category names
        # self.all_category_names = [cat["name"] for cat in self.coco_gt.dataset["categories"]]


    def process(self, data_batch, data_samples):
        """Convert predictions into COCO format before calling standard COCO processing."""
        updated_data_samples = []  # Create a new list to store updated data samples

        for data_sample in data_samples:
            # Retrieve correct image ID
            img_id = data_sample.get("img_id", None)


            # Extract predicted instances safely
            pred_instances = data_sample.get("pred_instances", None)
            if pred_instances is None:
                continue  # Skip if no predictions

            if "labels" not in pred_instances:
                print(f"Warning: No labels found for image ID {img_id}")
                continue  # Skip if there are no labels

            # Get text prompts used for inference
            text_prompt = data_sample.get("text", None)
            if text_prompt is None:
                print(f"Warning: No text prompts found for image ID {img_id}")
                continue

            mapped_labels = []
            for label_idx in pred_instances["labels"].tolist():  # ✅ Use dictionary access
                category_name = text_prompt[label_idx]
                category_id = self.global_prompt_to_index.get(category_name, -1)
                if category_id == -1:
                    warnings.warn(f"Category '{category_name}' not found in COCO categories.")

                mapped_labels.append(category_id)

            # ✅ Convert mapped labels to a PyTorch tensor
            pred_instances["labels"] = torch.tensor(mapped_labels, dtype=torch.int64, device='cuda')

            # ✅ Ensure it's updated in the data sample
            data_sample["pred_instances"] = pred_instances
            updated_data_samples.append(data_sample)  # Store the modified sample

        # Call the parent class's process method with the updated data samples
        super().process(data_batch, updated_data_samples)

