import os
import sys
import argparse
import ast
import cv2
import time
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path

FILE = Path(__file__).resolve() 
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# from hrnet.misc.visualization import draw_points_and_skeleton, joints_dict
# from hrnet.misc.utils import find_person_id_associations

from feeder import LoadStreams, LoadRTSPStreams
from detector import ActionRecognition


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="CAM") 
    # parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        # default = 'configs/posec3d/x3d_shallow_ntu60_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/pyskl/ckpt/'
                 'posec3d/slowonly_r50_ntu120_xsub/joint.pth'),
                # 'posec3d/x3d_shallow_ntu60_xsub/joint.pth'),
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    
    args = parser.parse_args()
    
    return args



def main():
    args = parse_opt()
    mode = args.mode
    cam_sources = [0]
    rtsp_sources = ['192.168.1.64:554']

    if mode.upper() == "CAM":
        source = cam_sources
        num_source = len(cam_sources)

    elif mode.upper() == "RTSP":
        source = rtsp_sources
        num_source = len(rtsp_sources)
    
    else:
        print("There is no source")
        sys.exit()


    # calib
    # if mode.upper() == "RTSP":
    #     mapx = np.load('./img_calib/mapx.npy')
    #     mapy = np.load('./img_calib/mapy.npy')
        
    # feeder
    img_queues = [mp.Queue(maxsize=1) for i in range(num_source)]
    
    if mode.upper() == "CAM":
        img_processes = [mp.Process(target=LoadStreams(source, idx).get_frame, args=(img_queues[idx], ), daemon = True) for idx in range(num_source)]
    
    elif mode.upper() == "RTSP":
        img_processes = [mp.Process(target=LoadRTSPStreams(source, idx).get_frame, args=(img_queues[idx], ), daemon = True) for idx in range(num_source)]
    
    for idx in range(num_source):
        img_processes[idx].start()

    # recognizer 
    recognizer =  ActionRecognition(args)
        
    try:
        while True:
            for idx in range(num_source):
                frame = img_queues[idx].get()
                result = recognizer.recognize(frame)
                # if mode.upper() == "RTSP":
                #     frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                # result, pts = detector.detect_skeleton(frame)
                cv2.imshow("result", result)
                cv2.waitKey(1)

                # cv2.imshow("result ", result)
                # cv2.waitKey(1)
                
    except KeyboardInterrupt:

        cv2.destroyAllWindows()
        sys.exit()


if __name__ == '__main__':
    
    
    main()
