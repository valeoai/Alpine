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

import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix
from scipy.spatial import QhullError
from scipy.spatial import cKDTree as KDTree
from .utils.box_fitting import fit_2d_box_modest
from scipy.sparse.csgraph import connected_components


class Alpine:
    """Class to perform the Alpine clustering

    Parameters
    ----------
    thing_indexes : List[int]
        List of indexes of the things, must match indexes given in the y array
    thing_bboxes : Dict[int, List[int]]
        Dictionary with the indexes of the things as keys and the bounding boxes as values
    k : int
        Number of neighbors to consider
    split : bool
        Whether to split the clusters or not using the box splitting scheme
    margin : float
        Margin to consider when splitting clusters

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
    
    References
    ----------
    Sautier, C., Puy, G., Boulch, A., Marlet, R., Lepetit, V., "Clustering is back: 
    Reaching state-of-the-art LiDAR instance segmentation without training".

    Examples
    --------
    >>> import numpy as np
    >>> from alpine import Alpine
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> thing_indexes = [0, 1]
    >>> thing_bboxes = {0: [2., 2.], 1: [2., 2.]}
    >>> alpine = Alpine(thing_indexes, thing_bboxes)
    >>> alpine.fit(X, y)
    >>> alpine.labels_
    array([1, 1, 2, 2])
    """

    def __init__(self, thing_indexes: List[int], thing_bboxes: Dict[int, List[float]], k:int=32, split:bool=False, margin:float=1.3):
        self.thing_indexes = thing_indexes
        for k, v in thing_bboxes.items():
            if v[1] > v[0]:
                thing_bboxes[k] = [v[1], v[0]]
        self.thing_bboxes = thing_bboxes
        self.k = k
        self.split = split
        self.margin = margin
        self.labels_ = None
        assert len(self.thing_indexes) == len(self.thing_bboxes)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Perform the Alpine clustering

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            The input point cloud in 2D euclidean space
        y : ndarray of shape (n_samples)
            The input semantic labels
        """
        X = X[:, :2]
        offset = 1
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        for object_class in self.thing_indexes:
            mask = y == object_class
            if (n:= mask.sum()) > 0:
                if n == 1:
                    # We need at least 2 points to cluster
                    self.labels_[mask] = offset
                    offset += 1
                    continue
                # Get the class's points
                pc_mask = X[mask]
                # Obtain the clusters
                clusters = self._clusterize(pc_mask, self.thing_bboxes[object_class][1], k=self.k)

                if self.split:
                    inst_mask = np.zeros(pc_mask.shape[0], dtype=np.int32)
                    for id in range(clusters[0]):
                        # For all found clusters, split them if they are too big
                        mask_cluster = clusters[1] == id
                        sub_clusters = self._split_cluster(pc_mask[mask_cluster],
                                                          self.thing_bboxes[object_class][1], 
                                                          self.thing_bboxes[object_class],
                                                          margin=self.margin, 
                                                          k=self.k)
                        inst_mask[mask_cluster] = sub_clusters[1] + offset
                        offset += sub_clusters[0]

                    self.labels_[mask] = inst_mask
                else:
                    self.labels_[mask] = clusters[1] + offset
                    offset += clusters[0]
        return self
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        """Perform the Alpine clustering and return the labels

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            The input point cloud in 2D euclidean space
        y : ndarray of shape (n_samples)
            The input semantic labels

        Returns
        -------
        labels : ndarray of shape (n_samples)
            The instance labels for each point in the dataset
        """
        self.fit(X, y)
        return self.labels_

    def _clusterize(self, pc, th, k, dist=None, neighbors=None):
        # Project the pc to 2D
        pc = pc[:, :2]
        # Set k to at most the maximum number of points -1
        k = min((k, pc.shape[0]-1))
        # Get kNN
        if neighbors is None:
            kdtree = KDTree(pc)
            dist, neighbors = kdtree.query(pc, k=k+1)
            dist, neighbors = dist[:, 1:], neighbors[:, 1:]
        # Build graph
        orig = np.vstack([np.arange(pc.shape[0]) for _ in range(neighbors.shape[1])]).T
        # Apply threshold
        weights = (dist < th).astype("float")
        W = csr_matrix(
            (weights.flatten(), (orig.flatten(), neighbors.flatten())), 
            shape=(pc.shape[0], pc.shape[0])
        )
        # Make graph non-oriented
        W = (W + W.T) / 2
        # Get connected components
        n, labels = connected_components(W, directed=False, return_labels=True)
        return n, labels

    def _split_cluster(self, pc, th, box_size, margin=1.3, k=32):
        # If the cluster is of 2 points, box_fitting won't work.
        if pc.shape[0] < 3:
            return 1, np.zeros(pc.shape[0], dtype=np.int32)
        try:
            # Apply box_fitting and find box size in lxw
            _, x, y, _ = fit_2d_box_modest(pc[:, :2])
            # check if the box is smaller than the average (with margin)
            if max(x, y) < box_size[0] * margin and min(x, y) < box_size[1] * margin:
                return 1, np.zeros(pc.shape[0], dtype=np.int32)
            else:
                # binary search to find biggest th that splits the cluster in boxes smaller than box_size
                dt = th/2
                current_th = dt
                k_ = min((k, pc.shape[0]-1))
                kdtree = KDTree(pc)
                dist, neighbors = kdtree.query(pc, k=k_+1)
                dist, neighbors = dist[:, 1:], neighbors[:, 1:]
                while dt > 1e-3:
                    # we are looking for a threshold that splits the cluster in 2
                    dt = dt/2
                    clusters = self._clusterize(pc, current_th, k, dist, neighbors)
                    if clusters[0] == 1:
                        current_th -= dt
                    elif clusters[0] == 2:
                        # apply recursively to both clusters
                        mask_1 = clusters[1] == 0
                        mask_2 = clusters[1] == 1
                        n, clusters[1][mask_1] = self._split_cluster(pc[mask_1], current_th, box_size,
                                                                    margin=margin, k=k)
                        n2, cl = self._split_cluster(pc[mask_2], current_th, box_size,
                                                    margin=margin, k=k)
                        clusters[1][mask_2] = cl + n
                        return n + n2, clusters[1]
                    else:
                        current_th += dt
        except QhullError:
            pass
        return 1, np.zeros(pc.shape[0], dtype=np.int32)
