# optimized_optical_flow.py
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


def get_output_folder():
    """결과를 저장할 폴더 경로 반환"""
    output_folder = os.path.join(os.getcwd(), "motion_analysis_results", "raw_data")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def optimized_optical_flow():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)

    if not cap.isOpened():
        print("카메라 오픈 실패")
        return

    # 첫 프레임 캡처
    ret, prev_frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        cap.release()
        return

    # 그레이스케일 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # 중앙 영역 마스크 생성
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.25)
    mask = np.ones((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 0, -1)

    # 저해상도 설정
    small_size = (160, 120)
    prev_small = cv2.resize(prev_gray, small_size)
    small_mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
    small_mask = (small_mask > 0).astype(np.uint8)

    # 시각화 설정
    hsv_mask = np.zeros_like(prev_frame)
    hsv_mask[..., 1] = 255
    avg_flow = np.zeros(2)

    # 운동 데이터 저장을 위한 변수
    motion_data = []
    start_time = time.time()
    last_save_time = time.time()
    frame_count = 0
    cumulative_angle = 0
    cumulative_distance = 0
    prev_angle = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time() - start_time

        # 영상 처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, small_size)

        # 광학 흐름 계산
        flow = cv2.calcOpticalFlowFarneback(
            prev_small, small_gray, None, 0.5, 2, 5, 1, 5, 1.1, 0
        )

        flow = cv2.resize(flow, (w, h)) * 4.0
        flow[mask == 0] = 0

        # 운동 분석
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        small_mag = cv2.resize(magnitude, small_size)
        valid_pixels = (small_mask == 1) & (small_mag > 0)

        if np.any(valid_pixels):
            mag_thresh = np.percentile(small_mag[valid_pixels], 90)
            strong_flows = flow[
                cv2.resize((small_mag >= mag_thresh).astype(np.uint8), (w, h)) > 0
            ]

            if len(strong_flows) > 0:
                avg_flow = np.nan_to_num(np.nanmean(strong_flows, axis=0), nan=0.0)

                # 각도 변화 계산 (도 단위)
                current_angle = np.degrees(np.arctan2(avg_flow[1], avg_flow[0]))
                angle_change = (current_angle - prev_angle + 180) % 360 - 180
                cumulative_angle += angle_change
                prev_angle = current_angle

                # 이동 거리 계산 (픽셀 단위)
                distance = np.linalg.norm(avg_flow)
                cumulative_distance += distance

        # 10프레임마다 데이터 저장
        if frame_count % 10 == 0:
            motion_data.append(
                {
                    "timestamp": current_time,
                    "frame": frame_count,
                    "angle_change": angle_change,
                    "distance": distance,
                    "avg_flow_x": avg_flow[0],
                    "avg_flow_y": avg_flow[1],
                }
            )
            cumulative_angle = 0
            cumulative_distance = 0

        # 시각화
        hsv_mask[..., 0] = angle * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        # 평균 이동 표시
        if not np.isnan(avg_flow).any():
            end_point = (
                int(center[0] + avg_flow[0] * 10),
                int(center[1] + avg_flow[1] * 10),
            )
            cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 2)

        # 분석 영역 표시
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
        info_text = (
            f"Angle: {cumulative_angle:.1f} deg, Dist: {cumulative_distance:.1f} px"
        )
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # FPS 표시
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # 결과 표시
        top_row = np.hstack((frame, flow_rgb))
        cv2.imshow("Optical Flow Analysis", top_row)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        prev_gray = gray.copy()
        prev_small = small_gray.copy()

    # 데이터 저장
    if motion_data:
        df = pd.DataFrame(motion_data)
        output_folder = get_output_folder()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"motion_data_{timestamp}.csv")
        df.to_csv(filename, index=False)
        print(f"\n운동 데이터가 {filename}에 저장되었습니다")
        return filename  # 저장된 파일 경로 반환

    cap.release()
    cv2.destroyAllWindows()
    return None


if __name__ == "__main__":
    optimized_optical_flow()
