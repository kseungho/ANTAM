import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_result_folder(category):
    result_folder = os.path.join("bugtracking_results", category, "results")
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def analyze_all_data(csv_path, category):
    timestamp = get_timestamp()
    result_folder = create_result_folder(category)
    df = pd.read_csv(csv_path)

    # 데이터 처리
    df["angle_change"] = (df["direction"].diff().fillna(0) + 180) % 360 - 180
    df["cumulative_angle"] = df["angle_change"].cumsum()

    # 1. 각도 변화 분석 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["angle_change"], "r-")
    plt.title("Angle Change Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Change (deg)")
    plt.grid(True)
    angle_path = os.path.join(result_folder, f"angle_analysis_{timestamp}.png")
    plt.savefig(angle_path)
    plt.close()

    # 2. 속도 분석 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["velocity"], "b-")
    plt.title("Velocity Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (px/s)")
    plt.grid(True)
    velocity_path = os.path.join(result_folder, f"velocity_analysis_{timestamp}.png")
    plt.savefig(velocity_path)
    plt.close()

    # 3. 이동 경로 분석
    plt.figure(figsize=(10, 10))
    plt.plot(df["pos_x"], df["pos_y"], "b-", alpha=0.5)
    plt.scatter(df["pos_x"].iloc[0], df["pos_y"].iloc[0], c="g", s=100, label="Start")
    plt.scatter(df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], c="r", s=100, label="End")
    plt.title("Movement Path Analysis")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    path_path = os.path.join(result_folder, f"path_analysis_{timestamp}.png")
    plt.savefig(path_path)
    plt.close()

    # 통계 계산
    stats = {
        "max_velocity": df["velocity"].max(),
        "avg_velocity": df["velocity"].mean(),
        "changes_over_15": (df["angle_change"].abs() > 15).sum(),
        "changes_over_30": (df["angle_change"].abs() > 30).sum(),
        "changes_over_60": (df["angle_change"].abs() > 60).sum(),
        "total_distance": np.sqrt(
            (df["pos_x"].iloc[-1] - df["pos_x"].iloc[0]) ** 2
            + (df["pos_y"].iloc[-1] - df["pos_y"].iloc[0]) ** 2
        ),
    }

    # 통계 저장
    stats_path = os.path.join(result_folder, f"result_stats_{timestamp}.txt")
    with open(stats_path, "w") as f:
        f.write("=== Tracking Result Statistics ===\n\n")
        f.write(f"Maximum Velocity: {stats['max_velocity']:.2f} px/s\n")
        f.write(f"Average Velocity: {stats['avg_velocity']:.2f} px/s\n")
        f.write(f"Total Distance: {stats['total_distance']:.2f} px\n\n")
        f.write("Angle Change Events:\n")
        f.write(f"> 15°: {stats['changes_over_15']} times\n")
        f.write(f"> 30°: {stats['changes_over_30']} times\n")
        f.write(f"> 60°: {stats['changes_over_60']} times\n")

    print("\n=== 분석 결과 ===")
    print(f"각도 분석: {angle_path}")
    print(f"속도 분석: {velocity_path}")
    print(f"경로 분석: {path_path}")
    print(f"통계 결과: {stats_path}")

    return {
        "angle_plot": angle_path,
        "velocity_plot": velocity_path,
        "path_plot": path_path,
        "stats": stats_path,
    }
