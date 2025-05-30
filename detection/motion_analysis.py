# motion_analysis.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def create_output_structure(output_folder, base_filename):
    """출력 폴더 구조 생성"""
    # 메인 결과 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 하위 폴더 구조 정의
    folder_structure = {
        "velocity": os.path.join(output_folder, "velocity_analysis"),
        "path": os.path.join(output_folder, "path_analysis"),
        "angle": os.path.join(output_folder, "angle_analysis"),
        "stats": os.path.join(output_folder, "statistics"),
    }

    # 모든 폴더 생성
    for folder in folder_structure.values():
        os.makedirs(folder, exist_ok=True)

    return folder_structure


def load_and_process_data(filename):
    """CSV 파일을 로드하고 데이터 처리"""
    df = pd.read_csv(filename)

    # 음수 값 처리 (distance 제외)
    df["angle_change"] = -df["angle_change"]
    df["avg_flow_x"] = -df["avg_flow_x"]
    df["avg_flow_y"] = -df["avg_flow_y"]

    # 속도 계산 (픽셀/초 단위)
    df["time_diff"] = df["timestamp"].diff().fillna(0)
    df["velocity"] = df["distance"] / df["time_diff"].replace(0, np.nan)

    return df


def save_velocity_changes(df, folders, base_filename):
    """속도 변화 그래프 저장"""
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["velocity"], "b-", linewidth=2)
    plt.title("Velocity Changes Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (pixels/second)")
    plt.grid(True)

    output_path = os.path.join(folders["velocity"], f"velocity_{base_filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved velocity chart to {output_path}")


def save_movement_path(df, folders, base_filename):
    """이동 경로 시각화 저장"""
    plt.figure(figsize=(10, 10))

    # 누적 이동 경로 계산
    df["cumulative_x"] = df["avg_flow_x"].cumsum()
    df["cumulative_y"] = df["avg_flow_y"].cumsum()

    # 이동 경로 그리기
    plt.plot(df["cumulative_x"], df["cumulative_y"], "b-", linewidth=1, alpha=0.7)
    plt.scatter(
        df["cumulative_x"],
        df["cumulative_y"],
        c=df["timestamp"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )

    # 시작점과 끝점 표시
    plt.scatter(
        df["cumulative_x"].iloc[0],
        df["cumulative_y"].iloc[0],
        c="green",
        s=100,
        label="Start",
    )
    plt.scatter(
        df["cumulative_x"].iloc[-1],
        df["cumulative_y"].iloc[-1],
        c="red",
        s=100,
        label="End",
    )

    # 중앙 영역 표시 (원형 ROI)
    center = (0, 0)
    radius = (
        np.sqrt(
            (df["cumulative_x"].max() - df["cumulative_x"].min()) ** 2
            + (df["cumulative_y"].max() - df["cumulative_y"].min()) ** 2
        )
        * 0.25
    )
    circle = Circle(center, radius, color="r", fill=False, linestyle="--", linewidth=2)
    plt.gca().add_patch(circle)

    plt.title("Movement Path Visualization")
    plt.xlabel("X displacement (pixels)")
    plt.ylabel("Y displacement (pixels)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.colorbar(label="Time (seconds)")

    output_path = os.path.join(folders["path"], f"path_{base_filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved movement path to {output_path}")


def save_angle_changes(df, folders, base_filename):
    """각도 변화 그래프 저장"""
    plt.figure(figsize=(12, 6))

    # 누적 각도 계산
    df["cumulative_angle"] = df["angle_change"].cumsum()

    plt.plot(df["timestamp"], df["cumulative_angle"], "r-", linewidth=2)
    plt.title("Cumulative Angle Changes Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)

    output_path = os.path.join(folders["angle"], f"angle_{base_filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved angle chart to {output_path}")


def analyze_and_save_motion_data(input_file, output_folder=None):
    """데이터 분석 및 그래프 저장 메인 함수"""
    try:
        # 입력 파일의 기본 이름 추출 (확장자 제외)
        base_filename = os.path.splitext(os.path.basename(input_file))[0]

        # 출력 폴더 구조 생성
        if output_folder is None:
            output_folder = os.path.join(
                os.path.dirname(input_file), "motion_analysis_results"
            )

        folders = create_output_structure(output_folder, base_filename)

        # 데이터 로드 및 처리
        df = load_and_process_data(input_file)

        # 그래프 생성 및 저장
        save_velocity_changes(df, folders, base_filename)
        save_movement_path(df, folders, base_filename)
        save_angle_changes(df, folders, base_filename)

        # 통계 정보 저장
        stats_file = os.path.join(folders["stats"], f"stats_{base_filename}.txt")
        with open(stats_file, "w") as f:
            f.write(f"=== Motion Data Summary for {base_filename} ===\n")
            f.write(f"Total movement time: {df['timestamp'].iloc[-1]:.2f} seconds\n")
            f.write(f"Total frames: {df['frame'].iloc[-1]}\n")
            f.write(f"Total distance: {df['distance'].sum():.2f} pixels\n")
            f.write(f"Average velocity: {df['velocity'].mean():.2f} pixels/second\n")
            f.write(f"Maximum velocity: {df['velocity'].max():.2f} pixels/second\n")
            f.write(f"Total rotation: {df['angle_change'].sum():.2f} degrees\n")

        print(f"\nAll analysis results saved to: {output_folder}")
        print("Folder structure:")
        for folder_name, folder_path in folders.items():
            print(f"- {folder_name}: {folder_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python motion_analysis.py <input_csv_file> [output_folder]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_and_save_motion_data(input_file, output_folder)
