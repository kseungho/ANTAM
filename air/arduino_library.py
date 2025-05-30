import time

import numpy as np
import serial


# 아두이노 초기화
def initialize_arduino(port="COM3", baudrate=9600, timeout=1):
    arduino = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2)  # Arduino 초기화 대기 시간
    return arduino


# 누적 거리 계산 함수 (매 프레임 mouse_x, mouse_y를 받는 구조)
def calculate_distance(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)  # 유클리드 거리 계산


# 신호 보내는 함수
def send_sign(arduino, mouse_positions, threshold=100.0):
    total_distance = 0.0
    prev_x, prev_y = mouse_positions[0]

    for x, y in mouse_positions[1:]:
        dist = calculate_distance(prev_x, prev_y, x, y)
        total_distance += dist
        prev_x, prev_y = x, y

        if total_distance >= threshold:
            arduino.write(b"1")  # 바이트 형태로 전송
            total_distance = 0.0  # 거리 초기화
