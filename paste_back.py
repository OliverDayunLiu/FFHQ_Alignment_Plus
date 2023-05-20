import os
import numpy as np
import argparse
import scipy.ndimage
import face_alignment
import glob
from natsort import natsorted
import multiprocessing
import cv2


def paste_back(args, aligned_img_path, original_img_path, dst_file, face_landmarks, transform_size=1024, enable_padding=True):

    aligned_img = cv2.imread(aligned_img_path)
    original_img = cv2.imread(original_img_path)
    original_img_copy = original_img.copy()

    ### Calculate the matrix used in original alignment. Then we inverse it. ###

    lm = face_landmarks
    lm_eye_left = lm[36: 42, :2]  # left-clockwise
    lm_eye_right = lm[42: 48, :2]  # left-clockwise
    lm_mouth_outer = lm[48: 60, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    #Shrink.
    shrink = int(np.floor(qsize / aligned_img.shape[0] * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(original_img.shape[1]) / shrink)), int(np.rint(float(original_img.shape[0]) / shrink)))
        original_img = cv2.resize(original_img, rsize, interpolation=cv2.INTER_CUBIC)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, original_img.shape[1]),
            min(crop[3] + border, original_img.shape[0]))
    if crop[2] - crop[0] < original_img.shape[1] or crop[3] - crop[1] < original_img.shape[0]:
        original_img = original_img[crop[1]:crop[3],crop[0]:crop[2]]
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - original_img.shape[1] + border, 0),
           max(pad[3] - original_img.shape[0] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        original_img = np.pad(original_img.astype(np.float32), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = original_img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        original_img += (scipy.ndimage.gaussian_filter(original_img, [blur, blur, 0]) - original_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        original_img += (np.median(original_img, axis=(0, 1)) - original_img) * np.clip(mask, 0.0, 1.0)
        original_img = np.clip(np.rint(original_img), 0, 255).astype(np.uint8)
        quad += pad[:2]

    quad = quad.astype(np.float32)

    # Transform.
    dst = np.array([
            [0, 0],
            [0, transform_size - 1], # max height
            [transform_size - 1, transform_size - 1],
            [transform_size-1, 0], # max width
        ], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    M_inverse = scipy.linalg.inv(M)

    ### Apply the inverse transform ###
    img_aligned_scaled_back = cv2.resize(aligned_img, (transform_size, transform_size), interpolation=cv2.INTER_CUBIC)
    aligned_img_pasted_back = cv2.warpPerspective(img_aligned_scaled_back, M_inverse, (original_img_copy.shape[1], original_img_copy.shape[0]), flags=cv2.INTER_CUBIC)

    white_mask = np.zeros(img_aligned_scaled_back.shape, dtype=np.float32)
    white_mask_pasted_back = cv2.warpPerspective(white_mask, M_inverse,
                                                  (original_img_copy.shape[1], original_img_copy.shape[0]),
                                                  flags=cv2.INTER_CUBIC)

    white_mask_pasted_back = np.clip(white_mask_pasted_back, 0, 1)
    kernel = np.ones((5, 5), np.uint8)
    white_mask_pasted_back_dilated = cv2.dilate(white_mask_pasted_back, kernel, iterations=1)
    white_mask_pasted_back_dilated = np.clip(white_mask_pasted_back_dilated, 0, 1)

    result = white_mask_pasted_back_dilated * aligned_img_pasted_back + (1 - white_mask_pasted_back_dilated) * original_img_copy
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save aligned image.
    cv2.imwrite(dst_file, result)


def paste_back_single_worker_process(worker_arg):

    args = worker_arg['args']
    filepaths = worker_arg['filepaths']

    for i in range(len(filepaths)):
        filepath = filepaths[i]
        face_landmarks = np.load(os.path.join(args.landmark_dir, os.path.basename(filepath).replace('.png','.npy').replace('.jpg','.npy')))
        paste_back(args, filepath, os.path.join(args.original_imgs, os.path.basename(filepath)),
                   os.path.join(args.dst, os.path.basename(filepath)),
                   face_landmarks[0], args.transform_size,
                        args.no_padding)
        print('%d/%d' % (i+1, len(filepaths)))


def main():
    parser = argparse.ArgumentParser(description='paste back aligned image to original image. Currently only supports one face per image.')
    parser.add_argument('--aligned_imgs', default='./aligned_images', help='directory of aligned images (.jpg and .png)')
    parser.add_argument('--original_imgs', default='./original_images', help='directory of original full images (.jpg and .png)')
    parser.add_argument('--dst', default='./paste_back', help='directory of output images')
    parser.add_argument('--landmark_dir', default='',
                        help='The landmark you used to perform face alignment.')
    parser.add_argument('--transform_size', default=1024, type=int, help='Set this to the same value that you used when doing face alignment.')
    parser.add_argument('--no_padding', action='store_false', help='Set this to the same value that you used when doing face alignment.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='if >0, will run in multiprocess mode to save significant amount of time.')
    args = parser.parse_args()


    if not os.path.exists(args.dst):
        os.mkdir(args.dst)

    filepaths = natsorted(glob.glob(os.path.join(args.aligned_imgs, '*.png')) + glob.glob(os.path.join(args.aligned_imgs, '*.jpg')))
    args.num_workers = min(len(filepaths), args.num_workers)

    if args.num_workers > 0:
        # Multiprocess setup
        workers_args = []
        for i in range(args.num_workers):
            workers_args.append({})
            workers_args[-1]['args'] = args
            workers_args[-1]['filepaths'] = []

        which_worker = -1
        for i in range(len(filepaths)):
            which_worker += 1
            if which_worker >= args.num_workers:
                which_worker = 0
            workers_args[which_worker]['filepaths'].append(filepaths[i])

        with multiprocessing.get_context('spawn').Pool() as pool:
            pool.map(paste_back_single_worker_process, workers_args)
    else:
        worker_arg = {}
        worker_arg['args'] = args
        worker_arg['filepaths'] = filepaths
        paste_back_single_worker_process(worker_arg)


if __name__ == '__main__':
    main()
