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
from glob import glob
from alpine import Alpine
from scipy.special import softmax
from evaluation.eval_pq import PanopticEval

THING_CLASSES = [1,2,3,4,5,6,7,8] # 1,2,6
CLASSES = {
  0: "unlabeled",
  1: "car",
  2: "bicycle",
  3: "motorcycle",
  4: "truck",
  5: "other-vehicle",
  6: "person",
  7: "bicyclist",
  8: "motorcyclist",
  9: "road",
  10: "parking",
  11: "sidewalk",
  12: "other-ground",
  13: "building",
  14: "fence",
  15: "vegetation",
  16: "trunk",
  17: "terrain",
  18: "pole",
  19: "traffic-sign"
}

LEARNING_MAP_INVERSE = {
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81,    # "traffic-sign"
}

mapper = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}

THINGS = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
STUFF = ['road', 'parking','sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign']

BBOX_DATASET = {1: [3.9, 1.6], 2: [1.76, 0.6], 3: [2.11, 0.77],  # nuscenes' boxes to fill in the unknown
        4: [6.93, 2.51], 5: [10.5, 2.94], 6: [0.8, 0.6], 7: [1.76, 0.6], 8: [2.11, 0.77]}
BBOX_WEB = {1: [4.4, 1.8], # car: https://www.motor1.com/news/707996/vehicles-larger-than-ever-usa-europe
            2: [1.75, 0.61], # bicycle: https://thebestbikelock.com/wp-content/uploads/2020/01/one-bike-average-size.gif
            3: [2.2, 0.95], # motorcycle: https://carparkjourney.wordpress.com/2013/07/16/what-is-the-average-size-of-a-motorbike/
            4: [10, 3], # truck
            5: [10, 3], # other-vehicle
            6: [0.94, 0.94], # person: RLSP arm span height: https://pubmed.ncbi.nlm.nih.gov/25063245/  average height in germany https://en.wikipedia.org/wiki/Average_human_height_by_country 179. We get 179*1.06/2
            7: [1.75, 0.61], # bicyclist: bicycle
            8: [2.2, 0.95], # motorcyclist: motorcycle
            }


if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path_dataset", type=str, help="Path to dataset", default="datasets/semantic_kitti/")
    parser.add_argument("--path_to_files", type=str, nargs='+', help="Path to the semantic predictions")
    parser.add_argument("--phase", help="Split of the dataset to apply ALPINE on", default="val", choices=["train", "val", "test"])
    parser.add_argument("--save_file", type=str, help="Directory to save the results", default=None)
    parser.add_argument("--split", action="store_true", help="Split clusters using bbox size")
    parser.add_argument("--dataset_based", action="store_true", help="Use the dataset-based bboxes")
    parser.add_argument('--margin', type=float, help="Margin in the bbox size when splitting", default=1.3)
    parser.add_argument('-k', "--neighbors", type=int, default=32, help="number of neighbors")
    args = parser.parse_args()
    print(f"Evaluating {args.path_to_files}")

    phase_scenes= {"train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], "val": [8], "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}
    if args.phase == "train":
        split = phase_scenes["train"]
    elif args.phase == "val":
        split = phase_scenes["val"]
    elif args.phase == "test":
        split = phase_scenes["test"]
    else:
        raise Exception(f"Unknown split {args.phase}")
    # Find all files
    dataset = []
    for i_folder in np.sort(split):
        dataset.extend(
            glob(
                os.path.join(
                    args.path_dataset,
                    "sequences",
                    str(i_folder).zfill(2),
                    "velodyne",
                    "*.bin",
                )
            )
        )
    dataset = np.sort(dataset)

    if args.save_file is not None:
        save_path = args.save_file
        for sequence in phase_scenes[args.phase]:
            os.makedirs(os.path.join(save_path, "sequences", str(sequence).zfill(2), "predictions"), exist_ok=True)

        save_file = True
    else:
        save_file = False

    BBOX = BBOX_DATASET if args.dataset_based else BBOX_WEB

    alpine = Alpine(THING_CLASSES, BBOX, k=args.neighbors, split=args.split, margin=args.margin)
    # --- Evaluation
    evaluator = PanopticEval(20, ignore=[0], min_points=50)
    for it, batch in enumerate(
        tqdm(dataset, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Get the point cloud
        pc = np.fromfile(
            batch,
            dtype=np.float32,
        )
        pc = pc.reshape((-1, 4))[:, :2]
        if args.phase != "test":
            # Get the labels
            label_inst = np.fromfile(
                batch.replace("velodyne", "labels")[:-3] + "label",
                dtype=np.uint32,
            )
            
            label_sem = label_inst & 0xFFFF  # delete high 16 digits binary
            label_sem = np.vectorize(mapper.__getitem__)(label_sem).astype(
                np.int32
            )

        # Get the predictions. Many different formats were used in this project, hence the many different procedures
        sem_pred = np.zeros((pc.shape[0], 20), dtype=float)
        for i, path in enumerate(args.path_to_files):
            if path == 'oracle':
                sem_pred = label_sem
            elif os.path.exists(pred_path := os.path.join(path, 'sequences', batch.split('sequences/')[1].replace('velodyne', 'predictions')[:-3] + "label.npz")):
                # In this case, the results are logits
                sem_pred[:, 1:] += softmax(np.load(pred_path)["logits"], 2).mean(0)
            elif os.path.exists(pred_path := os.path.join(path, f"{batch.rsplit('/sequences/', 1)[1].split('/', 1)[0]}_{batch.rsplit('/', 1)[1][:-3]}npy")):
                pred = np.load(pred_path)
                if len(pred.shape) == 1:
                    sem_pred = pred
                else:
                    sem_pred[:, 1:] += softmax(pred, 2).mean(0)[:, 1:]
            elif os.path.exists(pred_path := os.path.join(path, f"{batch.rsplit('/sequences/', 1)[1].split('/', 1)[0]}_{batch.rsplit('/', 1)[1][:-4]}_pred.npy")):
                sem_pred[:, 1:] += softmax(np.load(pred_path), 1)
            elif os.path.exists(pred_path := os.path.join(path, 'sequences', batch.split('sequences/')[1].replace('velodyne', 'predictions')[:-3] + "label")):
                pred = np.fromfile(pred_path, dtype=np.uint32)
                pred = pred & 0xFFFF  # delete high 16 digits binary
                ins = pred
                sem_pred = np.vectorize(mapper.__getitem__)(pred).astype(np.int32)
            else:
                raise Exception(f"File not found {pred_path}")
        
        if len(sem_pred.shape) == 2:
            sem_pred = np.argmax(sem_pred, axis=1)

        # Clustering
        inst_pred = alpine.fit_predict(pc, sem_pred)

        if args.phase != "test":
            # Add the frame to the evaluation
            evaluator.addBatch(sem_pred, inst_pred, label_sem, label_inst)
        if save_file:
            # prepare file in semantickitti format
            inv_pred = np.vectorize(LEARNING_MAP_INVERSE.__getitem__)(sem_pred)
            label = (inst_pred << 16) + inv_pred  # reconstruct full label
            label = label.astype(np.uint32)
            # Save result
            label_file = batch.rsplit('sequences/', 1)[1]
            label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
            label_file = os.path.join(save_path, "sequences", label_file)
            label.tofile(label_file)

    if args.phase != "test":
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
    else:
        print(f"Results saved to {save_path}")