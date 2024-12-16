import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法

cap = cv2.VideoCapture(0)

# mediapipe 啟用偵測全身
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(640,480))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = holistic.process(img2)              # 開始偵測全身
        #print(results)
        # 身體偵測，繪製身體骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        #landmarks = results.pose_landmarks.landmark
        #shoulder = [landmarks[mp_holistic.Holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.Holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        #print(shoulder)
        #np.save('skeleton_data.npy', results.pose_landmarks)
        #if results.left_hip_landmarks:
            #for index, landmarks in enumerate(results.left_hip_landmarks.landmark):
                #print(index, landmarks)
        #print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP])
        cv2.imshow('warmup', img)
        if cv2.waitKey(5) == ord('q'):
            break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()