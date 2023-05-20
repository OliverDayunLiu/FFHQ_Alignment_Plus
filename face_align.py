import os
import numpy as np
import argparse
import scipy.ndimage
#import PIL.Image
import face_alignment
import glob
from natsort import natsorted
import multiprocessing
import cv2


# Original PIL version.
'''
def image_align(args, src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]


        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if args.before_output_shape_dir != '':
            before_output_shape = img.size
            np.save(os.path.join(args.before_output_shape_dir, os.path.basename(src_file).replace('.png', '.npy').replace('.jpg', '.npy')), before_output_shape)

        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file)
'''


# Rewritten in CV because it lets you inverse perspective transformation easily. Neat for paste back script.
def image_align(args, src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
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

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = cv2.imread(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.shape[1]) / shrink)), int(np.rint(float(img.shape[0]) / shrink)))
        img = cv2.resize(img, rsize, interpolation=cv2.INTER_CUBIC)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.shape[1]),
            min(crop[3] + border, img.shape[0]))
    if crop[2] - crop[0] < img.shape[1] or crop[3] - crop[1] < img.shape[0]:
        img = img[crop[1]:crop[3],crop[0]:crop[2]]
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.shape[1] + border, 0),
           max(pad[3] - img.shape[0] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(img.astype(np.float32), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
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
    img = cv2.warpPerspective(img, M, (transform_size, transform_size), flags=cv2.INTER_CUBIC)

    if output_size < transform_size:
        img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    # Save aligned image.
    cv2.imwrite(dst_file, img)


def face_align_single_worker_process(worker_arg):

    args = worker_arg['args']
    filepaths = worker_arg['filepaths']
    gpu_id = worker_arg['gpu_id']

    if args.input_landmark_dir == '':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    for i in range(len(filepaths)):
        filepath = filepaths[i]
        if args.input_landmark_dir == '':
            face_landmarks = landmarks_detector.get_landmarks(filepath)
            if args.output_landmark_dir != '':
                np.save(os.path.join(args.output_landmark_dir, os.path.basename(filepath).replace('.png', '.npy').replace('.jpg', '.npy')), face_landmarks)
        else:
            face_landmarks = np.load(os.path.join(args.input_landmark_dir, os.path.basename(filepath).replace('.png','.npy').replace('.jpg','.npy')))
        if not args.allow_multiple_faces:
            image_align(args, filepath, os.path.join(args.dst, os.path.basename(filepath)), face_landmarks[0], args.output_size, args.transform_size,
                        args.no_padding)
        else:
            for j in face_landmarks.shape[0]:
                image_align(args, filepath, os.path.join(args.dst, os.path.basename(filepath).replace('.png', '_%2d.png' % j).replace('.jpg', '_%2d.jpg' % j)), face_landmarks[j], args.output_size,
                            args.transform_size,
                            args.no_padding)
        print('%d/%d' % (i+1, len(filepaths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='./raw_images', help='directory of raw images (.jpg and .png)')
    parser.add_argument('--dst', default='./aligned_images', help='directory of aligned images to output to')
    parser.add_argument('--input_landmark_dir', default='',
                        help='if provided, will skip landmark detection step and use the ones stored in this folder.')
    parser.add_argument('--output_landmark_dir', default='', help='If provided, will save detected landmarks to this folder. The shape for each file is num_faces x 68 x 2.')
    parser.add_argument('--output_size', default=512, type=int, help='size of aligned output')
    parser.add_argument('--transform_size', default=1024, type=int, help='size of aligned transform')
    parser.add_argument('--no_padding', action='store_false', help='no padding')
    parser.add_argument('--allow_multiple_faces', action='store_true',
                        help='If set, in cases where there are multiple faces in a single image, face alignment will be done on all of them.'
                             'If not set, we only perform face alignment on the first face found in the image.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='if >0, will run in multiprocess mode to save significant amount of time.')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus on your machine')
    args = parser.parse_args()


    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    if args.output_landmark_dir != '' and not os.path.exists(args.output_landmark_dir):
        os.mkdir(args.output_landmark_dir)

    filepaths = natsorted(glob.glob(os.path.join(args.src, '*.png')) + glob.glob(os.path.join(args.src, '*.jpg')))
    args.num_workers = min(len(filepaths), args.num_workers)

    if args.num_workers > 0:
        # Multiprocess setup
        workers_args = []
        for i in range(args.num_workers):
            workers_args.append({})
            workers_args[-1]['args'] = args
            workers_args[-1]['filepaths'] = []
            workers_args[-1]['gpu_id'] = i % args.num_gpus

        which_worker = -1
        for i in range(len(filepaths)):
            which_worker += 1
            if which_worker >= args.num_workers:
                which_worker = 0
            workers_args[which_worker]['filepaths'].append(filepaths[i])

        with multiprocessing.get_context('spawn').Pool() as pool:
            pool.map(face_align_single_worker_process, workers_args)
    else:
        worker_arg = {}
        worker_arg['args'] = args
        worker_arg['filepaths'] = filepaths
        worker_arg['gpu_id'] = 0
        face_align_single_worker_process(worker_arg)


if __name__ == '__main__':
    main()
