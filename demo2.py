# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from subprocess import call
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2
import os


parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

sourceDir = "examples/driving"

# ffmpeg -i out-image-%04d.jpg -c:v libx264 -vf "fps=30" out.mp4

# Flow Options:
alpha = 0.012
ratio = 0.5
minWidth = 30
nOuterFPIterations = 5
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

files = sorted(os.listdir(sourceDir))

last_filename = ""

for index, filename in enumerate(files):
    if last_filename == "":
        last_filename = filename
    else:
        im1 = np.array(Image.open(sourceDir + "/" + last_filename))
        im2 = np.array(Image.open(sourceDir + "/" + filename))
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        s = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        e = time.time()
        print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
            e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('result/out-' + last_filename, rgb)
        last_filename = filename
        # cv2.imwrite('examples/result/', im2W[:, :, ::-1] * 255)

print('Finished. Start converting files.')
os.chdir("result")

for index, filename in enumerate(files):
    print("Combining: ", filename, "out-" + filename)
    call(["magick", "convert", "../" + sourceDir + "/" + filename, "out-" + filename, "+append", "final-" + filename])

print("Finished. Start creating movie.")
finalName = "out.mp4"
call(["ffmpeg", "-i", "final-image-%04d.jpg", "-c:v", "libx264", "-vf", "fps=30", finalName]) 
print("Done.")
