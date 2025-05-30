import time

import cv2
import numpy as np

from tracking_utils import (
    adjust_points_order,
    calculate_angle_and_points,
    calculate_angle_between_points,
    calculate_center_of_mass,
    detect_closest_contour,
    initialize_camera,
)


def main():
    cap = initialize_camera()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame Rate: {fps} FPS")
    previous_point1 = None
    previous_point2 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                if point1 is not None and point2 is not None:
                    point1, point2 = adjust_points_order(
                        point1, point2, previous_point1, previous_point2
                    )
                    previous_point1, previous_point2 = point1, point2

                    vector_angle = calculate_angle_between_points(point1, point2)

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

        cv2.imshow("Original with Closest Contour", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
