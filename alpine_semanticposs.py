# Copyright (c) 2025 Valeo Comfort and Driving Assistance - Corentin Sautier @ valeo.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import argparse
import numpy as np
from tqdm import tqdm
from alpine import Alpine
from evaluation.eval_pq import PanopticEval


THING_CLASSES = [1, 2, 3]
CLASSES = {
  0: "unlabeled",
  1: "person",
  2: "rider",
  3: "car",
  4: "trunk",
  5: "plant",
  6: "traffic sign",
  7: "pole",
  8: "garbage-can",
  9: "building",
  10: "cone",
  11: "fence",
  12: "bicycle",
  13: "ground",
}

mapper = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 6, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 0, 19: 0, 20: 0, 21: 12, 22: 13}

THINGS = ['person', 'rider', 'car']
STUFF = ['trunk', 'plant', 'traffic sign', "pole", "garbage-can", "building", "cone", "fence", "bicycle", "ground"]
BBOX_DATASET = {1: [0.8, 0.6], 2: [1.76, 0.6], 3: [3.9, 1.6]}  # Copied from SK
BBOX_WEB = {1: [0.94, 0.94], 2: [1.75, 0.61], 3: [4.4, 1.8]}  # Copied from SK

if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path_dataset", type=str, help="Path to dataset", default="datasets/semantic_poss/")
    parser.add_argument("--path_to_files", type=str, help="Path to the semantic predictions")
    parser.add_argument("--phase", help="Split of the dataset to apply ALPINE on", default="val", choices=["train", "val"])
    parser.add_argument("--split", action="store_true", help="Split clusters using bbox size")
    parser.add_argument("--dataset_based", action="store_true", help="Use the web-based bboxes")
    parser.add_argument('--margin', type=float, help="Margin in the bbox size when splitting", default=1.3)
    parser.add_argument('-k', "--neighbors", type=int, default=32, help="number of neighbors")
    args = parser.parse_args()

    phase_scenes= {"train": [0, 1, 3, 4, 5], "val": [2]}

    dataset = []
    scenes = phase_scenes[args.phase]
    for scene in scenes:
        scene_path = os.path.join(args.path_dataset, "sequences", str(scene).zfill(2), "velodyne")
        dataset.extend([os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))])

    BBOX = BBOX_DATASET if args.dataset_based else BBOX_WEB

    alpine = Alpine(THING_CLASSES, BBOX, k=args.neighbors, split=args.split, margin=args.margin)
    # --- Evaluation
    evaluator = PanopticEval(14, ignore=[0], min_points=50, offset=2**31)
    for it, batch in enumerate(
        tqdm(dataset, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Get the point cloud
        pc = np.fromfile(
            batch,
            dtype=np.float32,
        )
        pc = pc.reshape((-1, 4))[:, :2]
        # Get the labels
        label_inst = np.fromfile(
            batch.replace("velodyne", "labels")[:-3] + "label",
            dtype=np.uint32,
        )
        # Filter eval classes.
        label_sem = label_inst & 0xFFFF  # delete high 16 digits binary
        label_sem = np.vectorize(mapper.__getitem__)(label_sem).astype(
            np.int32
        )

        # Get the predictions.
        scene = batch.rsplit('sequences/', 1)[1].split('/', 1)[0]
        if args.path_to_files == 'oracle':
            sem_pred = label_sem
        else:
            pred_path = os.path.join(args.path_to_files, f"{scene}_{batch.rsplit('/', 1)[1][:-4]}_pred.npy")
            sem_pred = np.load(pred_path) + 1

        # Clustering
        inst_pred = alpine.fit_predict(pc, sem_pred)

        # Add the frame to the evaluation
        evaluator.addBatch(sem_pred, inst_pred, label_sem, label_inst)

    # Get the results
    mean_pq, mean_sq, mean_rq, class_all_pq, class_all_sq, class_all_rq = evaluator.getPQ()
    mean_iou, class_all_iou = evaluator.getSemIoU()

    mean_pq, mean_sq, mean_rq, mean_iou = mean_pq.item(), mean_sq.item(), mean_rq.item(), mean_iou.item()
    class_all_pq = class_all_pq.flatten().tolist()
    class_all_sq = class_all_sq.flatten().tolist()
    class_all_rq = class_all_rq.flatten().tolist()
    class_all_iou = class_all_iou.flatten().tolist()

    results = dict()
    results["all"] = dict(PQ=mean_pq, SQ=mean_sq, RQ=mean_rq, mIoU=mean_iou)
    for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_pq, class_all_rq, class_all_sq, class_all_iou)):
        results[CLASSES[idx]] = dict(PQ=pq, SQ=sq, RQ=rq, IoU=iou)
    thing_pq_list = [float(results[c]["PQ"]) for c in THINGS]
    stuff_iou_list = [float(results[c]["IoU"]) for c in STUFF]
    results["all"]["PQ_dagger"] = np.mean(thing_pq_list + stuff_iou_list)

    ALL_CLASSES = THINGS + STUFF

    PQ_all = np.mean([float(results[c]["PQ"]) for c in ALL_CLASSES])
    PQ_dagger = np.mean([float(results[c]["PQ"]) for c in THINGS] + [float(results[c]["IoU"]) for c in STUFF])
    RQ_all = np.mean([float(results[c]["RQ"]) for c in ALL_CLASSES])
    SQ_all = np.mean([float(results[c]["SQ"]) for c in ALL_CLASSES])

    PQ_things = np.mean([float(results[c]["PQ"]) for c in THINGS])
    RQ_things = np.mean([float(results[c]["RQ"]) for c in THINGS])
    SQ_things = np.mean([float(results[c]["SQ"]) for c in THINGS])

    PQ_stuff = np.mean([float(results[c]["PQ"]) for c in STUFF])
    RQ_stuff = np.mean([float(results[c]["RQ"]) for c in STUFF])
    SQ_stuff = np.mean([float(results[c]["SQ"]) for c in STUFF])
    print(f"PQ: {PQ_all}, PQ_dagger: {PQ_dagger}, RQ: {RQ_all}, SQ: {SQ_all}")
    print(f"PQ_things: {PQ_things}, RQ_things: {RQ_things}, SQ_things: {SQ_things}")
    print(f"PQ_stuff: {PQ_stuff}, RQ_stuff: {RQ_stuff}, SQ_stuff: {SQ_stuff}")
