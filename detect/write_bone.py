import cv2
import mediapipe as mp
import numpy as np
skeleton_data_array = np.load('skeleton_data.npy',allow_pickle=True)
frame_index=0
cap = cv2.VideoCapture("小腿前伸.mp4")    # 加载视频
mp_pose = mp.solutions.pose    # 定义Pose模块
pose = mp_pose.Pose()    # 创建Pose实例
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height), isColor=True)
while True:
    ret, frame = cap.read()    # 读取视频帧
    if not ret:
        break
     # 进行骨架检测
    results = pose.process(frame)
 
     # 获取骨架信息
    landmarks = results.pose_landmarks
    
     # 如果检测到骨架，则显示骨架
    if landmarks is not None:
        # 声明一个黑色图像
        black_img = np.ones_like(frame)* 255
        # 绘制骨架
        mp.solutions.drawing_utils.draw_landmarks(black_img, landmarks, mp_pose.POSE_CONNECTIONS)
        # 显示骨架
        cv2.imshow("Skeleton", black_img)
        out.write(black_img)
    
     # 按下'q'键退出
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()    # 释放视频
cv2.destroyAllWindows()    # 关闭所有窗口