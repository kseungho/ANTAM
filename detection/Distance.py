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
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    return cap


def calculate_fps(frame_count, start_time, fps_history, max_fps_history=3):
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        fps_history.append(fps)
        max_fps_history = 5
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


def main():
    arduino = initialize_arduino()
    cap = initialize_camera()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FPS, 60)
    print(f"Frame Rate: {fps} FPS")
    previous_point1 = None
    previous_point2 = None
    frame_count = 0
    start_time = time.time()
    fps_history = []

    previous_point1 = None
    previous_point2 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        closest_contour, min_distance = detect_closest_contour(mask, center_x, center_y)

        if closest_contour is not None:
            cv2.drawContours(frame, [closest_contour], -1, (255, 255, 255), 1)
            cx, cy = calculate_center_of_mass(closest_contour)

            if cx is not None and cy is not None:
                origin_cx = cx - center_x
                origin_cy = cy - center_y
                c2cdistance = np.sqrt((origin_cx) ** 2 + (origin_cy) ** 2)

                point1, point2, angle = calculate_angle_and_points(
                    closest_contour, cx, cy
                )

                if point1 and point2:
                    point1, point2 = adjust_points_order(
                        point1, point2, previous_point1, previous_point2
                    )
                    previous_point1, previous_point2 = point1, point2

                    vector_x = point2[0] - point1[0]
                    vector_y = point2[1] - point1[1]
                    vector_angle = np.degrees(np.arctan2(-vector_y, vector_x))

                    if vector_angle < 0:
                        vector_angle += 360

                    send_angle_to_arduino(arduino, vector_angle)

                    cv2.putText(
                        frame,
                        f"Angle with X-axis: {vector_angle:.1f} degrees",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 155, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.circle(frame, point1, 5, (255, 0, 255), -1)
                    cv2.circle(frame, point2, 5, (255, 255, 0), -1)
                    cv2.line(frame, point1, point2, (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"C2C Dist: {c2cdistance:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (155, 0, 155),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Closest Dist: {min_distance:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 155),
                    2,
                    cv2.LINE_AA,
                )

        avg_fps = calculate_fps(frame_count, start_time, fps_history)
        if avg_fps is not None:
            cv2.putText(
                frame,
                f"Avg FPS: {avg_fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (155, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Original with Closest Contour", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
