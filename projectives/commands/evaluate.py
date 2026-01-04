import os
import json
import click
import exif
import tqdm
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

from ..utils import end, start, status

@click.group()
def _evaluate():
    pass

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_giou(poly1, poly2):
    # Convert to Shapely polygons
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    
    # Calculate intersection and union areas
    if not polygon1.intersects(polygon2):
        return 0.0
    
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    
    # Find the smallest enclosing box
    min_x = min(polygon1.bounds[0], polygon2.bounds[0])
    min_y = min(polygon1.bounds[1], polygon2.bounds[1])
    max_x = max(polygon1.bounds[2], polygon2.bounds[2])
    max_y = max(polygon1.bounds[3], polygon2.bounds[3])
    
    enclosing_area = (max_x - min_x) * (max_y - min_y)
    
    # Calculate GIoU
    iou = intersection / union
    giou = iou - (enclosing_area - union) / enclosing_area
    
    return giou

def find_best_matches_bitpartite(ground_truth, prediction):
    n_gt = len(ground_truth)
    n_pred = len(prediction)
    print("n_gt: ", n_gt)
    print("n_pred: ", n_pred)
    print("--"*42)

    # create cost matrix (GIoU)
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt_shape in enumerate(ground_truth):
        gt_points = gt_shape["points"]
        for j, pred_shape in enumerate(prediction):
            pred_points = pred_shape["points"]
            giou = calculate_giou(gt_points, pred_points)
            cost_matrix[i, j] = -giou # We want to maximize GIoU, so we use negative values for minimization

    # print(cost_matrix)
    # Apply Hungarian algorithm for "optimal" assignment
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        giou = -cost_matrix[gt_idx, pred_idx]  # Convert back to positive GIoU
        
        gt_label = ground_truth[gt_idx]["label"]
        pred_label = prediction[pred_idx]["label"]

        print("GT: ", gt_label)
        print("PRED: ", pred_label)
        print("-"*42)

        matches.append({   
            "ground_truth_idx": gt_idx,
            "prediction_idx": pred_idx,
            "giou": giou,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "match": gt_label == pred_label
        })
    return matches


@_evaluate.command()
@click.argument("gt_dir", type=click.Path(exists=True))
@click.argument("dir", type=click.Path(exists=True))
def evaluate(gt_dir , dir):
    start("Evaluating")
    gt_todo = [f for f in os.listdir(gt_dir) if f.endswith(".json")]
    todo = [f for f in os.listdir(dir) if f.endswith(".json")]

    global_correct_matches = 0
    global_total_matches = 0
    
    # for each todo file find the corresponding gt file of same name in gt_todo dir
    for file in tqdm.tqdm(todo):
        gt_file = [f for f in gt_todo if f == file]
        if len(gt_file) == 0:
            status(f"No ground truth file found for {file.name}")
            continue

        # load the json files
        gt_data = load_json(os.path.join(gt_dir, gt_file[0]))
        pred_data = load_json(os.path.join(dir, file))

        # Get the shape data
        gt_shapes = gt_data["shapes"] if isinstance(gt_data, dict) else Exception("Invalid JSON format for ground truth data. Expected a dictionary.")
        pred_shapes = pred_data["shapes"] if isinstance(pred_data, dict) else Exception("Invalid JSON format for ground truth data. Expected a dictionary.")

        # find the best matches 
        matches = find_best_matches_bitpartite(gt_shapes, pred_shapes)
        
        # Calculate accuracy
        correct_matches = sum(1 for match in matches if match["match"])
        total_matches = len(matches)
        accuracy = correct_matches / total_matches if total_matches > 0 else 0
        
        global_correct_matches += correct_matches
        global_total_matches += total_matches

        # print(f"Matched {total_matches} polygons")
        # print(f"Correct label matches: {correct_matches}")
        # print(f"Accuracy: {accuracy:.2%}")

    accuracy = global_correct_matches / global_total_matches if global_total_matches > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")
    


        

        