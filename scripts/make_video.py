import sys
import glob
from multiprocessing import Pool

import numpy as np
import skimage.io
from cv2 import cv2
import subprocess


def padding(array, xx, yy):
    h = array.shape[0]
    w = array.shape[1]
    a = max(xx - h, 0)
    b = max(yy - w, 0)
    return np.pad(array, pad_width=((0, a), (0, b), (0, 0)), mode='constant', constant_values=255)


def create_frame(args):
    idx, vid_dir, max_goals = args
    height = 960
    tree_width = 384

    road_image = skimage.io.imread(vid_dir + f'/road_{idx}.png')[:, :, :3]
    tree_files = glob.glob(vid_dir + f'/tree_{idx}_*.png')
    tree_files.sort()
    tree_images = [padding(skimage.io.imread(f)[:, :, :3], height, tree_width) for f in tree_files]
    if len(tree_images) > 0:
        stacked_tree_images = np.hstack(tree_images)
    else:
        stacked_tree_images = np.ones((812, tree_width * max_goals, 3)) * 255
    stacked_tree_images = padding(stacked_tree_images, height, tree_width * max_goals)

    rescale_factor = height / road_image.shape[0]
    road_image_scaled = cv2.resize(road_image, (int(road_image.shape[1] * rescale_factor), height))
    full_image = np.hstack([road_image_scaled, stacked_tree_images])
    skimage.io.imsave(vid_dir + f'/img{idx:03d}.png', full_image)


def main():
    workers = 8
    vid_dir = sys.argv[1]
    road_files = glob.glob(vid_dir + '/road_*.png')
    num_frames = len(road_files)

    max_goals = 0
    for idx in range(num_frames):
        tree_files = glob.glob(vid_dir + f'/tree_{idx}_*.png')
        if len(tree_files) > max_goals:
            max_goals = len(tree_files)

    args_list = []
    for idx in range(num_frames):
        args_list.append((idx, vid_dir, max_goals))

    with Pool(workers) as p:
        p.map(create_frame, args_list)


if __name__ == '__main__':
    main()
