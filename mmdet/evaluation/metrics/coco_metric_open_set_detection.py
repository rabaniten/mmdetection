import warnings
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
import torch

# ToDo: not not hardcode label here, check how to access them.
ALL_LABELS = (
    "transparent plate cover",
    "soup cover",
    "metallic plate cover",
    "rectangular metallic cover",
    "other cover",
    "plastic wrap",
    "cover that is above its tableware",
    "cover that occludes food",
)


@METRICS.register_module()
class OpenSetCOCOMetric(CocoMetric):
    """Custom COCO Evaluator for Open-Set Detection Models in MMDetection."""

    def __init__(self, ann_file, outfile_prefix=None, **kwargs):
        super().__init__(ann_file=ann_file, outfile_prefix=outfile_prefix, **kwargs)

        self.ann_file = ann_file
        print(
            f"🔍 OpenSetCOCOMetric initialized with annotation file: {self.ann_file}",
            flush=True,
        )

        # Load COCO ground truth annotations
        self.coco_gt = COCO(self.ann_file)

        # Mapping: category name → internal label index
        self.global_prompt_to_index = {name: idx for idx, name in enumerate(ALL_LABELS)}

        print(
            f"✅ Loaded {len(self.global_prompt_to_index)} categories from COCO annotations.",
            flush=True,
        )

    def process(self, data_batch, data_samples):
        """Convert predictions into COCO format before calling standard COCO processing."""
        updated_data_samples = []

        for data_sample in data_samples:
            img_id = data_sample.get("img_id", None)

            pred_instances = data_sample.get("pred_instances", None)
            if pred_instances is None:
                print(
                    f"⚠️  No pred_instances found for image ID {img_id}, skipping.",
                    flush=True,
                )
                continue

            if "labels" not in pred_instances:
                print(
                    f"⚠️  No 'labels' in pred_instances for image ID {img_id}, skipping.",
                    flush=True,
                )
                continue

            text_prompt = data_sample.get("text", None)
            if text_prompt is None:
                print(
                    f"⚠️  No 'text' found for image ID {img_id}, skipping.", flush=True
                )
                continue

            mapped_labels = []
            for label_idx in pred_instances["labels"].tolist():
                try:
                    category_name = text_prompt[label_idx]
                    category_id = self.global_prompt_to_index.get(category_name, -1)

                    if category_id == -1:
                        warnings.warn(
                            f"⚠️ Category '{category_name}' not found in COCO categories."
                        )

                    mapped_labels.append(category_id)

                except Exception as e:
                    print(
                        f"❌ Error processing label index {label_idx}: {e}", flush=True
                    )
                    continue

            pred_instances["labels"] = torch.tensor(
                mapped_labels, dtype=torch.int64, device="cuda"
            )
            data_sample["pred_instances"] = pred_instances
            updated_data_samples.append(data_sample)

        super().process(data_batch, updated_data_samples)
