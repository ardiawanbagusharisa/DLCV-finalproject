
import torch
import numpy as np
from utils.data_utils import load_datasets, create_detection_datasets, collate_fn
from utils.conditional_model import load_conditional_model
from utils.training_utils import compute_metrics
from transformers.trainer_utils import EvalPrediction

def debug():
    print("Loading datasets...")
    ds_train, ds_valid = load_datasets(
        coco_dir="E:/Python/DLCV/dataset/coco",
        coco_rain_dir="E:/Python/DLCV/dataset/coco_rain",
        coco_ratio=0.1, rain_ratio=0.1 # Small subset
    )
    
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model, processor = load_conditional_model(
        rain_detector_path="./rain_detector_pretrained/rain_detector_best.pt",
        spdnet_path="E:/Python/DLCV/Project DLCV/model_spa.pt",
        rtdetr_name="PekingU/rtdetr_r18vd",
        device=device
    )
    model.to(device) # Ensure entire model is on device
    
    print("Creating dataset...")
    _, valid_dataset = create_detection_datasets(ds_train, ds_valid, processor, percent_dataset=10)
    
    print("Getting one batch...")
    loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    # Move batch to device
    batch["pixel_values"] = batch["pixel_values"].to(device)
    if "pixel_mask" in batch:
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
    
    # Move labels to device
    new_labels = []
    for label in batch["labels"]:
        new_label = {}
        for k, v in label.items():
            if isinstance(v, torch.Tensor):
                new_label[k] = v.to(device)
            else:
                new_label[k] = v
        new_labels.append(new_label)
    batch["labels"] = new_labels
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**batch)
    
    print("Computing metrics...")
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    labels = batch["labels"]
    
    eval_pred = EvalPrediction(predictions=(logits, pred_boxes), label_ids=labels)
    
    # Call compute_metrics (which now has debug prints)
    metrics = compute_metrics(eval_pred, processor, threshold=0.05)
    print("Metrics:", metrics)

if __name__ == "__main__":
    debug()
