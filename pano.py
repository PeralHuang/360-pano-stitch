# -*- coding: utf-8 -*-
"""
Created on 2019-05-09 23:36:36
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
from stitch import Stitcher, Method

import sys

import cv2
import numpy as np


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


def show_imageA(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)).show()


def warp_cylindrical(img, f, x_c, y_c, shape):
    w_, h_ = shape

    y_i, x_i = np.indices((h_, w_))
    B = np.stack([f * np.tan((x_i - x_c) / f) + x_c, (y_i - y_c) / np.cos((x_i - x_c) / f) + y_c], axis=-1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_rgba = img
    return cv2.remap(img_rgba, B.astype(np.float32), None, cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)


def warp_spherical(img, f, x_c, y_c, shape):
    w_, h_ = shape
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    B = np.stack([f * np.tan((x_i - x_c) / f) + x_c, f * np.tan((y_i - y_c) / f) / np.cos((x_i - x_c) / f) + y_c],
                 axis=-1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return cv2.remap(img_rgba, B.astype(np.float32), None, cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)


def main():
    images = []
    images_cyl = []
    # images.append(cv2.imread('resource/pano/1.jpg'))
    # images.append(cv2.imread('resource/pano/2.jpg'))
    # images.append(cv2.imread('resource/pano/3.jpg'))
    for i in range(13):
        # f = 1130
        print(f'Reading {i}')
        # images.append(cv2.imread(f'resource/middle/middle_{i+1}.png'))
        images.append(cv2.imread(f'resource/cmu1/medium{(i)%13:0>2}.jpg'))
    h, w = images[0].shape[:2]
    f = 1130
    print(f"Focal length {f}")
    for i in images:
        print('Warping')
        images_cyl.append(warp_cylindrical(i, f, w / 2, h / 2, (w, h)))

    image_last = images_cyl[0]
    for index, image in enumerate(images_cyl):
        if index == 0:
            continue
        try:
            if image is None:
                print('File not exist')
                sys.exit(1)
            print(f'======\nStitching {index + 1}')
            stitcher = Stitcher(image, image_last, Method.SIFT, False, match_threshold=2)
            stitcher.stich(show_match_point=1, use_gauss_blend=0, show_result=0)
            image_last = stitcher.image
            cv2.imwrite('r:/temp/' + 'tmp-{}.png'.format(index), image_last)
        except Exception as e:
            print(f"Error on match {index + 1}:", e)
            continue

    cv2.imwrite('output/' + 'pano.png', image_last)
    # i_image = warp_cylindrical_inverse(image_last, f, w / 2, h / 2)
    # cv2.imwrite('output/' + 'pano_i.png', i_image)
    show_imageA(image_last)
    print("Done")


if __name__ == "__main__":
    main()