# -*- coding: utf-8 -*-
"""
Created on 2019-05-15 15:33:33
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
# from __future__ import annotations
import sys
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def warp_cylindrical(img, f, x_c, y_c, shape):
    w_, h_ = shape

    y_i, x_i = np.indices((h_, w_))
    B = np.stack([f * np.tan((x_i - x_c) / f) + x_c, (y_i - y_c) / np.cos((x_i - x_c) / f) + y_c], axis=-1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_rgba = img
    return cv2.remap(img_rgba, B.astype(np.float32), None, cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)


def transform_homography_keypoints(keypoints: List[cv2.KeyPoint], M) -> List[cv2.KeyPoint]:

    pt = []
    for k in keypoints:
        pt.append(k.pt)
    pt = np.array([pt])
    trans_kp = cv2.perspectiveTransform(pt, M)[0]
    res = []
    for j in trans_kp:
        res.append(cv2.KeyPoint(j[0], j[1], 1))
    return res


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


def show_imageA(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)).show()


class Method(Enum):

    SURF = cv2.xfeatures2d.SURF_create
    SIFT = cv2.xfeatures2d.SIFT_create
    ORB = cv2.ORB_create


class KeypointImage:
    def __init__(self, image, keypoints=None, descriptors=None, method=Method.SIFT):
        self.image = image
        self.keypoints = None
        self.descriptors = None
        self.method = method
        self.M = np.eye(3)
        self.f = None
        self.compute_keypoint()

    def show(self):
        show_image(self.image)

    def compute_keypoint(self, margin=100):
        # remove keypoing near the border
        feature = self.method.value(800)
        keypoints, descriptors = feature.detectAndCompute(self.image, None)
        kp = []
        des = []
        for i in range(len(keypoints)):
            x, y = keypoints[i].pt
            h, w = self.image.shape[:2]
            if x < margin or y < margin or x > w - margin or y > h - margin:
                pass
            else:
                kp.append(keypoints[i])
                des.append(descriptors[i])
        self.keypoints = kp
        self.descriptors = des

    def show_keypoint(self):
        if self.keypoints is None:
            self.compute_keypoint()
        out = cv2.drawKeypoints(self.image, self.keypoints, None, flags=4)
        show_image(out)

    def transform_cylindrical_keypoints(self, f, x_c, y_c) -> List[cv2.KeyPoint]:
        if self.keypoints is None:
            self.compute_keypoint()
        pt = []
        for k in self.keypoints:
            pt.append(k.pt)
        pt = np.array(pt)
        x_i = pt[:, 0]
        y_i = pt[:, 1]

        trans_kp = np.stack(
            [f * np.arctan2(x_i - x_c, f) + x_c, f * (y_i - y_c) / np.sqrt((x_i - x_c)**2 + f**2) + y_c], axis=-1)
        res = []
        for j in trans_kp:
            res.append(cv2.KeyPoint(j[0], j[1], 1))
        return res

    def warp_cylindrical(self, f=None, x_c=None, y_c=None, shape=None) -> 'KeypointImage':
        f = f or self.f
        h, w = shape or self.image.shape[:2]

        if x_c is None or y_c is None:
            x_c, y_c = w/2, h/2

        y_i, x_i = np.indices((h, w))
        B = np.stack([f * np.tan((x_i - x_c) / f) + x_c, (y_i - y_c) / np.cos((x_i - x_c) / f) + y_c], axis=-1)

        kp = self.transform_cylindrical_keypoints(f, x_c, y_c)
        img_rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        img_rgba = self.image
        return KeypointImage(
            cv2.remap(img_rgba, B.astype(np.float32), None, cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT),
            keypoints=kp,
            descriptors=self.descriptors)

    def transform_homography_keypoints(self, M=None) -> List[cv2.KeyPoint]:
        if self.keypoints is None:
            self.compute_keypoint()
        pt = []
        for k in self.keypoints:
            pt.append(k.pt)
        pt = np.array([pt])  # for opencv

        if M is None:
            M = self.M
        trans_kp = cv2.perspectiveTransform(pt, M)[0]
        res = []
        for j in trans_kp:
            res.append(cv2.KeyPoint(j[0], j[1], 1))
        return res

    def warp_homography(self, M=None, shape=None):
        M = M or self.M
        shape = shape or self.image.shape[:2]
        kp = self.transform_homography_keypoints(M)
        return KeypointImage(
            cv2.warpPerspective(self.image, M, shape),
            keypoints=kp,
            descriptors=self.descriptors)


class GaussianBlend:

    LEVEL = 6

    def __init__(self, image1: np.ndarray, image2: np.ndarray, mask: np.ndarray):
        self.image1 = image1
        self.image2 = image2
        if np.issubdtype(mask.dtype, np.integer):
            self.mask = mask / 255
        else:
            self.mask = mask

    def blend(self):
        print("Calculating pyramid")
        la1 = self.get_laplacian_pyramid(self.image1)
        la2 = self.get_laplacian_pyramid(self.image2)

        gm = self.get_gaussian_pyramid(self.mask)

        result = np.zeros(self.image1.shape, int)
        # new_la = []
        for i in range(self.LEVEL):
            mask = next(gm)
            # new_la.append(next(la1) * mask + next(la2) * (1.0 - mask))

            result += (next(la1) * mask + next(la2) * (1.0 - mask)).astype(int)
            # result += next(la1) + next(la2)
            del mask
            print(i, " level blended")
        return np.clip(result, 0, 255).astype('uint8')

        # print("Rebuilding ")
        # return self.rebuild_image(new_la)

    @classmethod
    def get_laplacian_pyramid(cls, image: np.ndarray):
        # output = []
        last = image

        for i in range(cls.LEVEL - 1):
            this = gaussian_filter(last, (3, 3, 0))
            laplace = cls.subtract(last, this)
            # output.append(laplace)
            yield laplace
            last = this
        # output.append(last)
        yield last
        # return output

    @classmethod
    def get_gaussian_pyramid(cls, image: np.ndarray):
        # G = []
        tmp = image
        for i in range(cls.LEVEL):
            # G.append(tmp)
            yield tmp
            tmp = gaussian_filter(tmp, (3, 3, 0))

        # return G

    @staticmethod
    def rebuild_image(laplacian_pyramid: np.ndarray):
        result = np.sum(laplacian_pyramid, axis=0)
        return np.clip(result, 0, 255).astype('uint8')

    @staticmethod
    def subtract(array1: np.ndarray, array2: np.ndarray):
        """give non minus subtract

        Args:
            array1 (np.ndarray): array1
            array2 (np.ndarray): array2

        Returns:
            np.ndarray: (array1 - array2)>0?(array1 - array2):0
        """
        array1 = array1.astype(int)
        array2 = array2.astype(int)
        result = array1 - array2
        # result[np.where(result < 0)] = 0

        return result  # .astype(np.uint8)


class Stitcher:
    def __init__(self, f):
        self.f = f
        self.shape = np.array([0, 0])
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.matcher = None
        self.result = None

    def add_image(self, image: KeypointImage):
        image.f = self.f
        if self.images == []:
            image = image.warp_cylindrical()
            self.result = image.image
            self.images.append(image)
            self.keypoints.extend(image.keypoints)
            self.descriptors.extend(image.descriptors)
            self.shape = np.maximum(self.shape, image.image.shape[:2])
            if image.method == Method.ORB:
                # error if not set this
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                # self.matcher = cv2.BFMatcher(crossCheck=True)
                self.matcher = cv2.FlannBasedMatcher()
            return
        # transformed_kp = image.transform_cylindrical_keypoints(self.f, image.image.shape[1]/2, image.image.shape[0]/2)
        image = image.warp_cylindrical()
        transformed_kp = image.keypoints

        try:
            match_points1, match_points2 = self.match(self.keypoints,
                                                      transformed_kp,
                                                      np.array(self.descriptors),
                                                      np.array(image.descriptors), image.image,
                                                      threshold=4)
        except Exception as e:
            print(f"Error on {e}, cannot find match")
            return
        M, _ = cv2.findHomography(match_points2, match_points1, method=cv2.RANSAC)
        print(M)
        # k = transform_homography_keypoints(transformed_kp, M)
        # self.keypoints = transformed_kp
        # self.descriptors = image.descriptors

        left, right, top, bottom = self.get_transformed_size(image.image.shape, M)
        # print(self.get_transformed_size())
        width = int(max(right, self.result.shape[1]) - min(left, 0))
        height = int(max(bottom, self.result.shape[0]) - min(top, 0))
        print(width, height)

        adjustM = np.array(
            [
                [1, 0, max(-left, 0)],  # 横向
                [0, 1, max(-top, 0)],  # 纵向
                [0, 0, 1]
            ],
            dtype=np.float64)
        # print('adjustM: ', adjustM)
        M = np.dot(adjustM, M)
        transformed_1 = cv2.warpPerspective(image.image, M, (width, height))
        transformed_2 = cv2.warpPerspective(self.result, adjustM, (width, height))
        self.result = self.blend_with_adjustion(transformed_1, transformed_2, M, adjustM)
        image.M = M  # TODO::
        self.descriptors.extend(image.descriptors)
        self.keypoints = transform_homography_keypoints(self.keypoints, adjustM)
        self.keypoints.extend(transform_homography_keypoints(transformed_kp, M))
        self.images.append(image)
        show_image(self.result)

    def blend_with_adjustion(self, image1: np.ndarray, image2: np.ndarray, M, adjustM) -> np.ndarray:
        mask = self.generate_mask(image1)
        show_image((mask * 255).astype('uint8'))

        # return (image1 * mask + image2 * (1 - mask)).astype('uint8')
        return self.gaussian_blend(image1, image2, mask)

    def gaussian_blend(self, image1: np.ndarray, image2: np.ndarray, mask: np.ndarray):

        return GaussianBlend(image1, image2, mask).blend()

    def generate_mask(self, image1: np.ndarray):
        print("Generating mask")
        mask = np.ones(image1.shape)
        mask = np.logical_and(mask, image1).astype(float)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        mask_blend = 10
        mask = gaussian_filter(mask.astype(float), (mask_blend, mask_blend, 0))
        mask = cv2.erode(mask, kernel, iterations=5)

        return mask

        # def get_transformed_position(pos, M):
        #     x, y = pos
        #     p = np.array([x, y, 1])[np.newaxis].T
        #     pa = np.dot(M, p)
        #     return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]
        # # x, y
        # center1 = image1.shape[1] / 2, image1.shape[0] / 2
        # center1 = get_transformed_position(center1, M)
        # center2 = image2.shape[1] / 2, image2.shape[0] / 2
        # center2 = get_transformed_position(center2, adjustM)
        # # 垂直平分线 y=-(x2-x1)/(y2-y1)* [x-(x1+x2)/2]+(y1+y2)/2
        # x1, y1 = center1
        # x2, y2 = center2

        # # note that opencv is (y, x)
        # def function(y, x, *z):
        #     return (y2 - y1) * y < -(x2 - x1) * (x - (x1 + x2) / 2) + (y2 - y1) * (y1 + y2) / 2

        # mask = np.fromfunction(function, image1.shape)

        # # mask = mask&_i2+mask&i1+i1&_i2
        # mask = np.logical_and(mask, np.logical_not(image2)) \
        #     + np.logical_and(mask, image1)\
        #     + np.logical_and(image1, np.logical_not(image2))

        # return mask

    def get_transformed_size(self, shape1, M) -> Tuple[int, int, int, int]:
        """计算形变后的边界
        计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """

        conner_0 = (0, 0)  # x, y
        conner_1 = (shape1[1], 0)
        conner_2 = (shape1[1], shape1[0])
        conner_3 = (0, shape1[0])
        points = [conner_0, conner_1, conner_2, conner_3]

        def get_transformed_position(pos):
            x, y = pos
            p = np.array([x, y, 1])[np.newaxis].T
            pa = np.dot(M, p)
            return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]

        # top, bottom: y, left, right: x
        top = min(map(lambda x: get_transformed_position(x)[1], points))
        bottom = max(map(lambda x: get_transformed_position(x)[1], points))
        left = min(map(lambda x: get_transformed_position(x)[0], points))
        right = max(map(lambda x: get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def match(self, keypoints1, keypoints2, descriptors1, descriptors2, image,  max_match_lenth=40, threshold=2):
        match_points = sorted(self.matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)

        match_len = min(len(match_points), max_match_lenth)

        # in case distance is 0
        if match_len < 4:
            print("Cannot find match")
            raise RuntimeError("找不到匹配")
            return None

        max_distance = threshold * match_points[0].distance

        for i in range(match_len):
            if match_points[i].distance > max_distance:
                match_len = i
                break
        assert(match_len >= 4)
        match_points = match_points[:match_len]
        print(f"Get {match_len} matches, min distance is {match_points[0].distance}, max distance is {match_points[-1].distance}")

        img3 = cv2.drawMatches(
            self.result, keypoints1, image, keypoints2, match_points, None, flags=0)
        show_image(img3)
        match_points1, match_points2 = [], []
        for i in match_points:
            match_points1.append(keypoints1[i.queryIdx].pt)
            match_points2.append(keypoints2[i.trainIdx].pt)

        match_points1 = np.float32(match_points1)
        match_points2 = np.float32(match_points2)
        return match_points1, match_points2


def read_image(image_file, method=Method.SIFT):
    return KeypointImage(cv2.imread(image_file), method=method)


def main():
    images = []
    # f = 1140
    f = 1150
    stitcher = Stitcher(f)
    # for i in range(13):
    #     # f = 1130
    #     print(f'Reading {i}')
    #     # images.append(cv2.imread(f'resource/middle/middle_{i+1}.png'))
    #     # img = read_image(f'resource/middle/middle_{((i+1)//2 * (-1)**(i%2))%14 +1}.png', method=Method.ORB)
    #     img = read_image(f'resource/middle/middle_{((i+1)//2 * (-1)**(i%2) + 6)%13}.png', method=Method.ORB)
    #     # images.append()
    #     stitcher.add_image(img)
    for i in range(13):
        # f = 1130
        print(f'Reading {i}')
        # images.append(cv2.imread(f'resource/middle/middle_{i+1}.png'))
        img = read_image(f'resource/cmu1/medium{((i+1)//2 * (-1)**(i%2))%13:0>2}.jpg', method=Method.SIFT)
        # images.append()
        stitcher.add_image(img)

    cv2.imwrite('output/' + 'pano.png', stitcher.result)


def test_wrap_kp():
    img = KeypointImage(cv2.imread('resource\CMU1\medium00.jpg'))
    # img.show()
    # img.show_keypoint()
    # img.show_keypoint()
    f = 1150
    h, w = img.image.shape[:2]

    img.compute_keypoint()
    kp = img.keypoints
    k1 = cv2.drawKeypoints(img.image, kp, None)
    k2 = warp_cylindrical(k1, f, w / 2, h / 2, (w, h))
    # show_image(k2)

    trans_kp = img.transform_cylindrical_keypoints(f, w / 2, h / 2)
    kps = []
    for j in trans_kp:
        kps.append(cv2.KeyPoint(j[0], j[1], 3))
    k3 = img.warp_cylindrical(f, w / 2, h / 2, (w, h))
    # show_image(k3)
    k4 = cv2.drawKeypoints(k3, kps, None)
    k5 = cv2.drawKeypoints(k3, kp, None)
    show_image(k4)
    show_image(k5)


if __name__ == "__main__":
    main()
