import time

import cv2
import numpy as np
import serial


def initialize_arduino(port="COM3", baudrate=9600, timeout=1):
    arduino = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2)  # Arduino initialization delay
    return arduino


def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    return cap


def calculate_fps(frame_count, start_time, fps_history, max_fps_history=3):
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        fps_history.append(fps)
        if len(fps_history) > max_fps_history:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        return avg_fps
    return None


def detect_closest_contour(mask, center_x, center_y):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_distance = float("inf")
    closest_contour = None

    for contour in contours:
        if cv2.contourArea(contour) > 10:
            for point in contour:
                x, y = point[0]
                origin_x = x - center_x
                origin_y = y - center_y
                distance = np.sqrt((origin_x) ** 2 + (origin_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour
    return closest_contour, min_distance


def calculate_center_of_mass(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None, None


def calculate_angle_and_points(contour, cx, cy):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, axes, angle) = ellipse
        length = axes[1] / 2
        x_offset = int(length * np.sin(np.radians(angle)))
        y_offset = int(length * np.cos(np.radians(angle)))
        point1 = (int(cx - x_offset / 2), int(cy + y_offset / 2))
        point2 = (int(cx + x_offset / 2), int(cy - y_offset / 2))
        return point1, point2, angle
    return None, None, None


def adjust_points_order(point1, point2, previous_point1, previous_point2):
    if previous_point1 is not None and previous_point2 is not None:
        dist1_to_prev1 = np.linalg.norm(np.array(point1) - np.array(previous_point1))
        dist1_to_prev2 = np.linalg.norm(np.array(point1) - np.array(previous_point2))
        if dist1_to_prev2 < dist1_to_prev1:
            point1, point2 = point2, point1
    return point1, point2


def send_angle_to_arduino(arduino, angle):
    if not hasattr(arduino, "last_sent") or time.time() - arduino.last_sent > 1:
        if 0 <= angle < 90:
            arduino.write(b"1\n")
        elif 90 <= angle < 180:
            arduino.write(b"2\n")
        elif 180 <= angle < 270:
            arduino.write(b"3\n")
        elif 270 <= angle < 360:
            arduino.write(b"4\n")
        arduino.last_sent = time.time()


def calculate_angle_between_points(point1, point2):
    """
    Calculate the angle between two points relative to the X-axis.
    """
    vector_x = point2[0] - point1[0]
    vector_y = point2[1] - point1[1]
    angle = np.degrees(np.arctan2(-vector_y, vector_x))
    if angle < 0:
        angle += 360
    return angle
