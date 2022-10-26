import cv2
# from queue import Queue
import multiprocessing as mp
# import sys
# import numpy as np

##### for single-process 
# class LoadStreams:
#     def __init__(self, source,i=0):
#         self.que = Queue(1)
#         self.cap = cv2.VideoCapture(int(source))

#     def get_frame(self, i=0):
#         assert self.cap.isOpened(), f'Failed to open cam {self.source}'

#         # cv2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
#         # cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
#         # cv2.set(cv2.CAP_PROP_FPS, 30)
        
       
#         ret, frame = self.cap.read()
        
#         if not ret:
#             print(f"Webcam {i} load failed")
#             sys.exit()
        
#         if self.que.empty():
#             self.que.put(frame)

#         return self.que


##### for multi-process
class LoadStreams:
    def __init__(self, source, i=0):
        self.source = source[i]

    def get_frame(self, queue:mp.Queue):
        cap = cv2.VideoCapture(self.source)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("RTLS image load failed")
                cv2.destroyAllWindows()
                cap.release()
                continue

            if queue.empty():
                queue.put(frame)

        
        cv2.destroyAllWindows()
        cap.release()


##### for single-process
# class LoadRTSPStreams:
#     def __init__(self, source, i=0):
#         self.que = Queue(1)
#         src = 'rtsp://admin:kongtech141219!@' + source[i] + '/Streaming/channels/101'
#         self.cap = cv2.VideoCapture(src)
        
#     def get_frame(self):
#         assert self.cap.isOpened(), f'Failed to open cam {self.source}'

#         ret, frame = self.cap.read()

#         if not ret:
#             print("RTLS image load failed")
#             cv2.destroyAllWindows()
#             self.cap.release()
#             sys.exit()

#         if self.que.empty():
#             self.que.put(frame)

#         return self.que



##### for multi-processing
class LoadRTSPStreams:
    def __init__(self, source, i=0):
        self.source = 'rtsp://admin:kongtech141219!@' + source[i] + '/Streaming/channels/101'
        
    def get_frame(self, queue:mp.Queue):
        cap = cv2.VideoCapture(self.source)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("RTLS image load failed")
                cv2.destroyAllWindows()
                cap.release()
                continue

            if queue.empty():
                queue.put(frame)

        
        cv2.destroyAllWindows()
        cap.release()