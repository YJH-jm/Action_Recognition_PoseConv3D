import sys
import os
import os.path as osp
import shutil
import time
import warnings

import cv2
import mmcv
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')




FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (0, 0, 0)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class ActionRecognition:
    def __init__(self, args):
        self.args = args
        self.config = mmcv.Config.fromfile(self.args.config)
        self.config.data.test.pipeline = [x for x in self.config.data.test.pipeline if x['type'] != 'DecompressPose']
        
        # Dafault -> config.model.type : Recognizer3D
        # Are we using GCN for Infernece?
        self.GCN_flag = 'GCN' in self.config.model.type
        if self.GCN_flag:
            format_op = [op for op in self.config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
            self.GCN_nperson = format_op['num_person']

        # Load Model
        self.model = init_recognizer(self.config, self.args.checkpoint, self.args.device)
        self.det_model = init_detector(self.args.det_config, self.args.det_checkpoint, self.args.device)
        assert self.det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                                'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
        assert self.det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
        
        # self.sk_model = init_pose_model(self.args.pose_config, self.args.pose_checkpoint, self.args.device)
        self.pose_model = init_pose_model(self.args.pose_config, self.args.pose_checkpoint, self.args.device)
        # Load label mae (label_class)
        # label_map = [x.strip() for x in open(args.label_map).readlines()]
        self.label_map = [x.strip() for x in open(args.label_map, 'r', encoding="UTF-8").readlines()] # Action Label Map 

    def detection_inference(self, frame):
        """Detect human boxes given frame paths.

        Args:
            frame_paths (list[str]): The paths of frames to do detection inference.

        Returns:
            list[np.ndarray]: The human detection results.
        """
        # print(args.det_config) # demo/faster_rcnn_r50_fpn_2x_coco.py
        # print(args.det_checkpoint) # http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
        # print(args.device) # cuda:0

        
        results = []
        # print('Performing Human Detection for each frame')

        
        result = inference_detector(self.det_model, frame) # result : list 
        # print("result 길이 : ", len(result))   # 80
        # print("result[0].shape : ", result[0].shape) # (2, 5) 
        # print("type(result[0]) : ", type(result[0])) #  <class 'numpy.ndarray'>
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= self.args.det_score_thr]
        results.append(result)

        return results

    def pose_inference(self, frame, det_results):
        # print("args.pose_config : ", args.pose_config) #  demo/hrnet_w32_coco_256x192.py
        # print("args.pose_checkpoint : ", args.pose_checkpoint) #  https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
        
        ret = []
        # print('Performing Human Pose Estimation for each frame')
        
        for f, d in zip(frame, det_results):
            # Align input format 
            # d (N, 5) - 각 사람에 대한 좌표와 c
            d = [dict(bbox=x) for x in list(d)] # [{"bbox : "}, {"bbox" : }] # N명의 bbox 존재
            pose = inference_top_down_pose_model(self.pose_model, f, d, format='xyxy')[0] #  [{"bbox : ", "keypoints" : }, {"bbox" : }]
            ret.append(pose)

        return ret

    def dist_ske(self, ske1, ske2):
        dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
        diff = np.abs(ske1[:, 2] - ske2[:, 2])
        return np.sum(np.maximum(dist, diff))

    def pose_tracking(self, pose_results, max_tracks=2, thre=30):
        tracks, num_tracks = [], 0
        num_joints = None
        for idx, poses in enumerate(pose_results):
            if len(poses) == 0:
                continue
            if num_joints is None:
                num_joints = poses[0].shape[0]
            track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
            n, m = len(track_proposals), len(poses)
            scores = np.zeros((n, m))

            for i in range(n):
                for j in range(m):
                    scores[i][j] = self.dist_ske(track_proposals[i]['data'][-1][1], poses[j])

            row, col = linear_sum_assignment(scores)
            for r, c in zip(row, col):
                track_proposals[r]['data'].append((idx, poses[c]))
            if m > n:
                for j in range(m):
                    if j not in col:
                        num_tracks += 1
                        new_track = dict(data=[])
                        new_track['track_id'] = num_tracks
                        new_track['data'] = [(idx, poses[j])]
                        tracks.append(new_track)
        tracks.sort(key=lambda x: -len(x['data']))
        result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
        for i, track in enumerate(tracks[:max_tracks]):
            for item in track['data']:
                idx, pose = item
                result[i, idx] = pose
        return result[..., :2], result[..., 2]

    def recognize(self, frame):
        h, w, _ =  frame.shape
        num_frame = 1
        # Get Human detection results
        t1 = time_sync()
        det_results = self.detection_inference(frame) # [numpy.ndarray_frame1, numpy.ndarray_frame2, ... ]
        if det_results[0].shape[0] == 0:
            return frame
        print("detection 된 사람 수 : ", det_results[0].shape[0])
        t2 = time_sync()
        print("det time : ", t2-t1)
        torch.cuda.empty_cache()
        pose_results = self.pose_inference([frame], det_results)  # [[{"bbox : ", "keypoints" : }, {"bbox" : }], frame2, .... ]
        print("skeleton 추출된 사람 수 : ", len(pose_results[0]))
        torch.cuda.empty_cache()
        t3 = time_sync()
        print("find skeleton time : ", t3-t2)
        
        fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

        if self.GCN_flag:
            # We will keep at most `GCN_nperson` persons per frame.
            tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
            keypoint, keypoint_score = self.pose_tracking(tracking_inputs, max_tracks=self.GCN_nperson)
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score
        else:
            num_person = max([len(x) for x in pose_results])
            # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
            num_keypoint = 17
            keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                dtype=np.float16)
            keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                    dtype=np.float16)
            for i, poses in enumerate(pose_results): # frame_idx, [{"bbox" : , "keypoint" : }, {"bbox" : , "keypoint" : }, .. 한 프레임에 있는 사람 수 만큼..] 
                for j, pose in enumerate(poses): # 한 프레임에 있는 사람 idx, 
                    pose = pose['keypoints']
                    keypoint[j, i] = pose[:, :2]
                    keypoint_score[j, i] = pose[:, 2]
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

        results = inference_recognizer(self.model, fake_anno)
       
        t4 = time_sync()
        print("pose recog time : ", t4-t3)

        action_label = self.label_map[results[0][0]]
        # print(args.pose_config) # 0 sdemo/hrnet_w32_coco_256x192.py
    
        vis_frames = [
            vis_pose_result(self.pose_model, frame, pose_results[i])
            for i in range(num_frame)
        ]
        for frame in vis_frames:
            cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

        t5 = time_sync()
        print('Pose vis time : ', t5-t4)
        return frame
        
        # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
        # vid.write_videofile(self.args.out_filename, remove_temp=True)

        # tmp_frame_dir = osp.dirname(frame_paths[0])
        # shutil.rmtree(tmp_frame_dir)