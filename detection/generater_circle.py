import math
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def generate_optimized_circles():
    a4_width, a4_height = 210, 297
    a4_area = a4_width * a4_height

    circle_radius = float(input("Enter the circle radius (mm): "))
    target_ratio = float(input("Enter the target area ratio (1-90%): ")) / 100
    target_color = float(input("1=red 2=green 3=blue :"))
    circle_area = math.pi * (circle_radius**2)
    max_theoretical = int((a4_area * target_ratio) / circle_area)
    initial_target = int(max_theoretical * 0.95)

    print(f"\n Theoretical max circles: {max_theoretical}")
    print(f" Initial target circles: {initial_target}")

    def place_circles(target_count):
        circles = []
        attempts = 0
        max_attempts = target_count * 5

        while len(circles) < target_count and attempts < max_attempts:
            x = np.random.uniform(circle_radius, a4_width - circle_radius)
            y = np.random.uniform(circle_radius, a4_height - circle_radius)

            if all(
                math.hypot(x - cx, y - cy) >= (r + circle_radius)
                for cx, cy, r in circles
            ):
                circles.append((x, y, circle_radius))
            attempts += 1

        return circles

    best_circles, best_ratio = [], 0
    tolerance = 0.01

    for attempt in range(3):
        circles = place_circles(initial_target)
        actual_ratio = (len(circles) * circle_area) / a4_area
        error = abs(actual_ratio - target_ratio)

        print(
            f"\nAttempt {attempt + 1}: Circles placed: {len(circles)}, Actual ratio: {actual_ratio * 100:.2f}%, Error: {error * 100:.2f}%"
        )

        if error < abs(best_ratio - target_ratio):
            best_circles, best_ratio = circles, actual_ratio

        if error <= tolerance:
            print("Target error range achieved")
            break

        initial_target = (
            int(initial_target * (1 + (target_ratio - actual_ratio) / 2))
            if actual_ratio < target_ratio
            else int(initial_target * 0.95)
        )
        initial_target = min(initial_target, max_theoretical)

    final_circles = best_circles
    final_ratio = best_ratio
    final_count = len(final_circles)
    if target_color == 1:
        filename = f"{int(round(target_ratio * 100))}%ratio_{int(circle_radius)}mm_redcircles_on_a4.pdf"
    elif target_color == 2:
        filename = f"{int(round(target_ratio * 100))}%ratio_{int(circle_radius)}mm_greencircles_on_a4.pdf"
    elif target_color == 3:
        filename = f"{int(round(target_ratio * 100))}%ratio_{int(circle_radius)}mm_bluecircles_on_a4.pdf"
    filename = filename.replace(" ", "_")

    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(a4_width / 25.4, a4_height / 25.4))
        ax.set_xlim(0, a4_width)
        ax.set_ylim(0, a4_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        for x, y, r in final_circles:
            if target_color == 1:
                ax.add_patch(
                    patches.Circle(
                        (x, y), r, facecolor="Red", edgecolor="white", alpha=0.7
                    )
                )
            elif target_color == 2:
                ax.add_patch(
                    patches.Circle(
                        (x, y), r, facecolor="Green", edgecolor="white", alpha=0.7
                    )
                )
            elif target_color == 3:
                ax.add_patch(
                    patches.Circle(
                        (x, y), r, facecolor="Blue", edgecolor="white", alpha=0.7
                    )
                )

        # 중앙에 반지름 124mm의 검정색 원 추가 (테두리 두께 2mm)
        center_x, center_y = a4_width / 2, a4_height / 2
        ax.add_patch(
            patches.Circle(
                (center_x, center_y),
                124,
                facecolor="none",
                edgecolor="black",
                linewidth=2,
            )
        )

        # 중앙에 반지름 10mm의 하얀색 원 추가
        ax.add_patch(
            patches.Circle(
                (center_x, center_y),
                10,
                facecolor="white",
                edgecolor="Yellow",
                linewidth=1,
            )
        )

        info_text = f"Radius: {circle_radius} mm\nRatio: {final_ratio * 100:.1f}%"
        plt.text(
            a4_width / 2,
            a4_height - 4,
            info_text,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.title("Circle Placement on A4", y=1.01)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    print(
        f"\n Final Results \nFilename: {filename}\nCircles placed: {final_count}\nRequested ratio: {target_ratio * 100:.2f}%\nActual ratio: {final_ratio * 100:.2f}%\nError: {abs(final_ratio - target_ratio) * 100:.2f}%\n"
    )
    print(f"File successfully created: {os.path.abspath(filename)}")


if __name__ == "__main__":
    print("=== A4 Optimized Circle Generator ===")
    generate_optimized_circles()
