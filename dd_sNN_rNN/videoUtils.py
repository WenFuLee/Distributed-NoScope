import cv2
import numpy as np
from math import ceil

def videoIterator(video_fname, scale=None, interval=1, start=0):
    cap = cv2.VideoCapture(video_fname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    frame = 0
    frame_ind = -1
    resol = None

    if scale is not None:
        try:
            len(scale)
            resol = scale
            scale = None
        except:
            resol = None

    while frame is not None:
        frame_ind += 1
        _, frame = cap.read()
        if frame_ind % interval != 0:
            continue
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        elif resol:
            frame = cv2.resize(frame, resol, interpolation=cv2.INTER_NEAREST)
        yield frame_ind, frame

def get_all_frames(num_frames, video_fname, scale=None, interval=1, start=0, dtype='float32'):

    true_num_frames = int(ceil((num_frames + 0.0) / interval))
    print('%d total frames / %d frame interval = %d actual frames' % (num_frames, interval, true_num_frames))
    vid_it = videoIterator(video_fname, scale=scale, interval=interval, start=start)

    _, frame = vid_it.__next__()
    frames = np.zeros( tuple([true_num_frames] + list(frame.shape)), dtype=dtype )
    frames[0, :] = frame

    for i in range(1, true_num_frames):
        _, frame = vid_it.__next__()
        frames[i, :] = frame

    if dtype == 'float32':
        frames /= 255.0

    return frames

