import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


class BackgroundMotionTracker:
    def __init__(self, camera_index=1, output_dir="motion_data"):
        # 카메라 초기화
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, 120)

        # 첫 프레임 캡처
        ret, self.prev_frame = self.cap.read()
        if not ret:
            raise RuntimeError("can't open camera")

        self.prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        self.hsv_mask = np.zeros_like(self.prev_frame)
        self.hsv_mask[..., 1] = 255

        # 광학 흐름 파라미터
        self.feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # 이동 경로 저장
        self.motion_history = []
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.prev_avg_angle = None

        # 출력 디렉토리
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 마스크 초기화
        self.prev_mask = np.ones_like(self.prev_gray, dtype=np.uint8) * 255
        h, w = self.prev_gray.shape
        cv2.rectangle(
            self.prev_mask,
            (w // 2 - 15, h // 2 - 15),
            (w // 2 + 15, h // 2 + 15),
            0,
            -1,
        )

        # 검정색 물체 검출 범위
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])

    def calculate_motion(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        current_time = time.time() - self.start_time
        time_interval = current_time - self.prev_time
        self.prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 검정색 물체 검출
        black_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        contours, _ = cv2.findContours(
            black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            cv2.circle(self.prev_mask, (int(x), int(y)), int(radius * 1.5), 0, -1)

        # 광학 흐름 계산
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=self.prev_mask, **self.feature_params
        )

        if prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None, **self.lk_params
            )

            if next_pts is not None:
                good_old = prev_pts[status == 1]
                good_new = next_pts[status == 1]

                if len(good_old) > 0:
                    motion_vectors = good_new - good_old
                    avg_motion = np.mean(motion_vectors, axis=0)

                    # 1차원/2차원 배열 모두 처리
                    if avg_motion.ndim == 1:
                        dx, dy = avg_motion[0], avg_motion[1]
                    else:
                        dx, dy = avg_motion[0, 0], avg_motion[0, 1]

                    speed = np.linalg.norm([dx, dy])
                    angle = np.degrees(np.arctan2(dy, dx)) % 360

                    # 각도 변화율 계산
                    angle_change = 0
                    if self.prev_avg_angle is not None:
                        diff = (angle - self.prev_avg_angle + 180) % 360 - 180
                        angle_change = diff / time_interval

                    self.prev_avg_angle = angle

                    # 데이터 저장
                    self.motion_history.append(
                        {
                            "timestamp": current_time,
                            "speed": speed,
                            "angle": angle,
                            "angle_change": angle_change,
                            "dx": dx,
                            "dy": dy,
                        }
                    )

        self.prev_gray = gray.copy()
        return True

    def save_results(self):
        df = pd.DataFrame(self.motion_history)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"motion_data_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"데이터가 {csv_path}에 저장되었습니다")

    def run(self):
        try:
            while True:
                if not self.calculate_motion():
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.save_results()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BackgroundMotionTracker(camera_index=1)  # 카메라 인덱스 변경 가능
    tracker.run()
