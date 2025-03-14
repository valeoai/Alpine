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
from scipy.special import softmax
from evaluation.eval_pq import PanopticEval


THING_CLASSES = [1,2,3,4,5,6,7,8,9,10] # 1,2,3,4,5,6,7,8,9,10,
CLASSES = {
    0: "ignore",
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "construction_vehicle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
    11: "driveable_surface",
    12: "other_flat",
    13: "sidewalk",
    14: "terrain",
    15: "manmade",
    16: "vegetation"
}
mapper = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0}
THINGS = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']
STUFF = ['terrain', 'vegetation', 'driveable_surface', 'other_flat', 'manmade', 'sidewalk']
BBOX_DATASET = {1: [2.53, 0.50], 2: [1.70, 0.60], 3: [10.5, 2.94], 4: [4.63, 1.97], 5: [6.37, 2.85], 6: [2.11, 0.77], 7: [0.73, 0.67], 8: [0.41, 0.41], 9: [12.29, 2.90], 10: [6.93, 2.51]}
BBOX_WEB = {1: [2., 0.5], # barrier: inferred
        2: [1.75, 0.61], # bicycle: https://thebestbikelock.com/wp-content/uploads/2020/01/one-bike-average-size.gif
        3: [10, 3], # bus: assuming bus, constr_veh, truck and trailer are 3x10m
        4: [4.75, 1.92], # car: https://www.finn.com/en-DE/campaign/supersized
        5: [10, 3], # construction_vehicle: assuming bus, constr_veh, truck and trailer are 3x10m
        6: [2.2, 0.95], # motorcycle: https://carparkjourney.wordpress.com/2013/07/16/what-is-the-average-size-of-a-motorbike/
        7: [0.93, 0.93], # person: RLSP arm span height: https://pubmed.ncbi.nlm.nih.gov/25063245/  average height in germany https://en.wikipedia.org/wiki/Average_human_height_by_country 175. We get 175*1.06/2
        8: [0.4, 0.4], # traffic_cone: found on the internet that cones are ~40cm large at the bottom
        9: [10, 3], # trailer: assuming bus, constr_veh, truck and trailer are 3x10m
        10: [10, 3], # truck: assuming bus, constr_veh, truck and trailer are 3x10m
        }

if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path_dataset", type=str, help="Path to dataset", default="datasets/nuscenes/")
    parser.add_argument("--path_to_files", type=str, nargs='+', help="Path to the semantic predictions")
    parser.add_argument("--phase", help="Split of the dataset to apply ALPINE on", default="val", choices=["train", "val", "test"])
    parser.add_argument("--save_file", type=str, help="Directory to save the results", default=None)
    parser.add_argument("--split", action="store_true", help="Split clusters using bbox size")
    parser.add_argument("--dataset_based", action="store_true", help="Use the dataset-based bboxes")
    parser.add_argument('--margin', type=float, help="Margin in the bbox size when splitting", default=1.3)
    parser.add_argument('-k', "--neighbors", type=int, default=32, help="number of neighbors")
    args = parser.parse_args()
    print(f"Evaluating {args.path_to_files}")

    current_folder = os.path.dirname(os.path.realpath(__file__))

    # List all keyframes
    dataset = np.load(
        os.path.join("evaluation", "list_files_nuscenes.npz")
    )[args.phase]
    if args.phase == "train":
        assert len(dataset) == 28130
    elif args.phase == "val":
        assert len(dataset) == 6019
    elif args.phase == "test":
        assert len(dataset) == 6008
    else:
        raise ValueError(f"Unknown phase {args.phase}.")

    # dataset = NuScenes(rootdir=args.path_dataset, phase=args.phase)
    if args.save_file is not None:
        save_path = os.path.join(args.save_file, 'panoptic', args.phase)
        os.makedirs(save_path, exist_ok=True)

        save_file = True
    else:
        save_file = False

    BBOX = BBOX_DATASET if args.dataset_based else BBOX_WEB

    alpine = Alpine(THING_CLASSES, BBOX, k=args.neighbors, split=args.split, margin=args.margin)
    # --- Evaluation
    evaluator = PanopticEval(17, ignore=[0], min_points=15)
    for it, batch in enumerate(
        tqdm(dataset, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Get the point cloud
        pc = np.fromfile(
            os.path.join(args.path_dataset, batch[0]),
            dtype=np.float32,
        )
        pc = pc.reshape((-1, 5))[:, :2]
        if args.phase != "test":
            # Get the labels
            panoptic_file = batch[1].replace("lidarseg", "panoptic").replace("bin", "npz")
            panoptic_label = np.load(os.path.join(args.path_dataset, panoptic_file))['data']
            # Filter eval classes.
            label_sem = np.vectorize(mapper.__getitem__)(panoptic_label // 1000)
            label_inst = panoptic_label

        # Get the predictions. Many different formats were used in this project, hence the many different procedures
        sem_pred = np.zeros((panoptic_label.shape[0], 16), dtype=float)
        for path in args.path_to_files:
            if path == "oracle":
                sem_pred = label_sem
            elif os.path.exists(pred_path := os.path.join(path, batch[1].rsplit('/', 1)[1].replace('lidarseg.bin', 'pred.npy'))):
                pred = np.load(pred_path)
                pred = pred / pred.sum(1, keepdims=True)
                sem_pred += pred
            elif os.path.exists(pred_path := os.path.join(path, batch[1].rsplit('/', 1)[1]+'.npz')):
                sem_pred += softmax(np.load(pred_path)["logits"], 2).mean(0)
            elif os.path.exists(pred_path := os.path.join(path, 'panoptic', args.phase, batch[2]+'_panoptic.npz')):
                sem_pred = np.load(pred_path)["data"] // 1000
            elif os.path.exists(pred_path := os.path.join(path, batch[1].rsplit('/', 1)[1])):
                sem_pred = np.fromfile(pred_path, np.uint8)
            else:
                raise Exception(f"File not found {pred_path}")
        
        if len(sem_pred.shape) == 2:
            sem_pred = np.argmax(sem_pred, axis=1) + 1

        # Clustering
        inst_pred = alpine.fit_predict(pc, sem_pred)

        if args.phase != "test":
            # Add the frame to the evaluation
            evaluator.addBatch(sem_pred, inst_pred, label_sem, label_inst)
        if save_file:
            # prepare file in nuscenes format
            panoptic_pred = (sem_pred * 1000 + inst_pred).astype(np.int32)
            np.savez(data=panoptic_pred, file=os.path.join(args.save_file, 'panoptic', args.phase, batch[1].rsplit('/', 1)[1].replace('_lidarseg.bin', '_panoptic.npz')))
    
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