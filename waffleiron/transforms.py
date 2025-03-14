# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# File adapted from https://github.com/valeoai/WaffleIron/blob/master/utils/transforms.py

import torch
import numpy as np


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, pcloud, labels):
        for t in self.transformations:
            pcloud, labels = t(pcloud, labels)
        return pcloud, labels


class RandomApply:
    def __init__(self, transformation, prob=0.5):
        self.prob = prob
        self.transformation = transformation

    def __call__(self, pcloud, labels):
        if torch.rand(1) < self.prob:
            pcloud, labels = self.transformation(pcloud, labels)
        return pcloud, labels


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels):
        if labels is None:
            return pcloud if self.inplace else np.array(pcloud, copy=True)

        out = (
            (pcloud, labels)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True))
        )
        return out


class Identity(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace)

    def __call__(self, pcloud, labels):
        return super().__call__(pcloud, labels)


class Rotation(Transformation):
    def __init__(self, dim=2, range=np.pi, inplace=False):
        super().__init__(inplace)
        self.range = range
        self.inplace = inplace
        if dim == 2:
            self.dims = (0, 1)
        elif dim == 1:
            self.dims = (0, 2)
        elif dim == 0:
            self.dims = (1, 2)
        elif dim == 6:
            self.dims = (4, 5)

    def __call__(self, pcloud, labels):
        # Build rotation matrix
        theta = (2 * torch.rand(1)[0] - 1) * self.range
        # Build rotation matrix
        rot = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Apply rotation
        pcloud, labels = super().__call__(pcloud, labels)
        pcloud[:, self.dims] = pcloud[:, self.dims] @ rot
        return pcloud, labels


class Scale(Transformation):
    def __init__(self, dims=(0, 1), range=0.05, inplace=False):
        super().__init__(inplace)
        self.dims = dims
        self.range = range

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        scale = 1 + (2 * torch.rand(1).item() - 1) * self.range
        pcloud[:, self.dims] *= scale
        return pcloud, labels


class FlipXY(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        id = torch.randint(2, (1,))[0]
        pcloud[:, id] *= -1.0
        return pcloud, labels


class LimitNumPoints(Transformation):
    def __init__(self, dims=(0, 1, 2), max_point=30000, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.max_points = max_point
        self.random = random
        assert max_point > 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if pc.shape[0] > self.max_points:
            if self.random:
                center = torch.randint(pc.shape[0], (1,))[0]
                center = pc[center : center + 1, self.dims]
            else:
                center = np.zeros((1, len(self.dims)))
            idx = np.argsort(np.square(pc[:, self.dims] - center).sum(axis=1))[:self.max_points]
            pc, labels = pc[idx], labels[idx]
        return pc, labels


class Crop(Transformation):
    def __init__(self, dims=(0, 1, 2), fov=((-5, -5, -5), (5, 5, 5)), eps=1e-4):
        super().__init__(inplace=True)
        self.dims = dims
        self.fov = fov
        self.eps = eps
        assert len(fov[0]) == len(fov[1]), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)

        where = None
        for i, d in enumerate(self.dims):
            temp = (pc[:, d] > self.fov[0][i] + self.eps) & (
                pc[:, d] < self.fov[1][i] - self.eps
            )
            where = temp if where is None else where & temp

        return pc[where], labels[where]


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if self.voxel_size <= 0:
            return pc, labels

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], labels[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind]