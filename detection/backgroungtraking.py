# main.py
import os
import time

from motion_analysis import analyze_and_save_motion_data
from optimized_optical_flow import optimized_optical_flow


def main():
    try:
        # 1. 첫 번째 프로그램 실행 (optical flow 분석)
        print("=== Running Optical Flow Analysis ===")
        csv_file = optimized_optical_flow()

        if not csv_file:
            print("No CSV file generated from optical flow analysis")
            return

        # 2. 두 번째 프로그램 실행 (motion data 분석)
        print("\n=== Running Motion Data Analysis ===")
        analyze_and_save_motion_data(csv_file)

        print("\n=== Analysis Complete ===")
        print(f"Raw data: {csv_file}")
        print(f"Analysis results in: {os.path.dirname(os.path.dirname(csv_file))}")

    except Exception as e:
        print(f"\nError in main program: {str(e)}")


if __name__ == "__main__":
    main()
