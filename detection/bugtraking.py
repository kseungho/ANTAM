import math
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SimpleBugTracker:
    def __init__(self):
        self.output_dir = "tracking_results"  # 結果を保存するディレクトリ
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # タイムスタンプ
        os.makedirs(self.output_dir, exist_ok=True)  # ディレクトリ作成

    def calculate_direction(self, dx, dy):
        """移動方向を計算 (0~360度)"""
        angle = math.degrees(math.atan2(dy, dx))  # 角度計算
        return angle % 360  # 0-360度に正規化

    def run_tracking(self):
        """トラッキング実行とデータ保存"""
        csv_path = os.path.join(
            self.output_dir, f"tracking_data_{self.timestamp}.csv"
        )  # CSVファイルパス

        # カメラ初期化
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 幅640ピクセル
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高さ480ピクセル
        cap.set(cv2.CAP_PROP_FPS, 120)  # フレームレート120fps

        tracking_data = []  # トラッキングデータ格納用
        start_time = time.time()  # 開始時間
        prev_pos, prev_time = None, start_time  # 前回位置と時間

        print("=== 黒色物体トラッキング開始 ===")
        print("カメラの前に黒色物体を置いてください")
        print("終了するにはQキーを押してください\n")

        while True:
            ret, frame = cap.read()  # フレーム取得
            if not ret:
                break  # フレーム取得失敗時

            current_time = time.time()  # 現在時刻

            # 黒色物体検出
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV色空間に変換
            mask = cv2.inRange(
                hsv, np.array([0, 0, 0]), np.array([180, 255, 50])
            )  # 黒色範囲マスク
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,  # 輪郭検出
            )

            current_pos = None  # 現在位置初期化
            if contours:
                largest = max(contours, key=cv2.contourArea)  # 最大輪郭取得
                M = cv2.moments(largest)  # モーメント計算
                if M["m00"] > 0:
                    current_pos = (
                        int(M["m10"] / M["m00"]),
                        int(M["m01"] / M["m00"]),
                    )  # 重心計算
                    cv2.circle(frame, current_pos, 7, (0, 255, 0), -1)  # 中心点描画

            # 0.1秒ごとにデータ記録
            if current_pos and (current_time - prev_time >= 0.1 or prev_pos is None):
                velocity, direction = 0, 0  # 速度と方向初期化
                if prev_pos:
                    dx = current_pos[0] - prev_pos[0]  # X方向移動量
                    dy = current_pos[1] - prev_pos[1]  # Y方向移動量
                    velocity = math.sqrt(dx**2 + dy**2) / (
                        current_time - prev_time
                    )  # 速度計算
                    direction = self.calculate_direction(dx, dy)  # 方向計算

                # データ追加
                tracking_data.append(
                    {
                        "timestamp": round(
                            current_time - start_time, 3
                        ),  # タイムスタンプ
                        "pos_x": current_pos[0],  # X座標
                        "pos_y": current_pos[1],  # Y座標
                        "velocity": round(velocity, 2),  # 速度
                        "direction": round(direction, 1),  # 方向
                    }
                )
                prev_pos, prev_time = current_pos, current_time  # 前回値更新

            # 画面表示
            cv2.imshow("Object Tracking (Press Q to stop)", frame)  # フレーム表示
            if cv2.waitKey(30) & 0xFF == ord("q"):  # Qキーで終了
                break

        cap.release()  # カメラ解放
        cv2.destroyAllWindows()  # ウィンドウ閉じる

        if tracking_data:
            pd.DataFrame(tracking_data).to_csv(csv_path, index=False)  # CSV保存
            print(f"\nトラッキングデータ保存完了: {csv_path}")
            return csv_path
        return None

    def analyze_data(self, csv_path):
        """データ分析と可視化"""
        df = pd.read_csv(csv_path)  # CSV読み込み

        # データ処理
        df["angle_change"] = (
            df["direction"].diff().fillna(0) + 180
        ) % 360 - 180  # 角度変化量計算

        # 1. 角度変化グラフ
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["angle_change"], "r-")  # 赤線でプロット
        plt.title("角度変化分析")  # タイトル
        plt.xlabel("時間 (秒)")  # X軸ラベル
        plt.ylabel("角度 (度)")  # Y軸ラベル
        angle_path = os.path.join(
            self.output_dir, f"angle_analysis_{self.timestamp}.png"
        )  # 保存パス
        plt.savefig(angle_path)  # 画像保存
        plt.close()

        # 2. 速度グラフ
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["velocity"], "b-")  # 青線でプロット
        plt.title("速度分析")  # タイトル
        plt.xlabel("時間 (秒)")  # X軸ラベル
        plt.ylabel("速度 (px/s)")  # Y軸ラベル
        velocity_path = os.path.join(
            self.output_dir, f"velocity_analysis_{self.timestamp}.png"
        )  # 保存パス
        plt.savefig(velocity_path)  # 画像保存
        plt.close()

        # 3. 移動経路可視化
        plt.figure(figsize=(10, 10))
        plt.plot(df["pos_x"], df["pos_y"], "b-", alpha=0.5)  # 経路線
        plt.scatter(
            df["pos_x"].iloc[0], df["pos_y"].iloc[0], c="g", s=100, label="開始点"
        )  # 開始点
        plt.scatter(
            df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], c="r", s=100, label="終了点"
        )  # 終了点
        plt.title("移動経路分析")  # タイトル
        plt.legend()  # 凡例表示
        path_path = os.path.join(
            self.output_dir, f"path_analysis_{self.timestamp}.png"
        )  # 保存パス
        plt.savefig(path_path)  # 画像保存
        plt.close()

        # 統計計算
        stats = {
            "max_velocity": df["velocity"].max(),  # 最大速度
            "avg_velocity": df["velocity"].mean(),  # 平均速度
            "changes_15": (df["angle_change"].abs() > 15).sum(),  # 15度以上変化回数
            "changes_30": (df["angle_change"].abs() > 30).sum(),  # 30度以上変化回数
            "changes_60": (df["angle_change"].abs() > 60).sum(),  # 60度以上変化回数
            "total_distance": np.sqrt(  # 総移動距離
                (df["pos_x"].iloc[-1] - df["pos_x"].iloc[0]) ** 2
                + (df["pos_y"].iloc[-1] - df["pos_y"].iloc[0]) ** 2
            ),
        }

        # 統計結果保存
        stats_path = os.path.join(self.output_dir, f"result_{self.timestamp}.txt")
        with open(stats_path, "w") as f:
            f.write("=== 最終分析結果 ===\n\n")
            f.write(f"最大速度: {stats['max_velocity']:.2f} px/s\n")
            f.write(f"平均速度: {stats['avg_velocity']:.2f} px/s\n")
            f.write(f"総移動距離: {stats['total_distance']:.2f} px\n\n")
            f.write("角度変化イベント:\n")
            f.write(f"15°以上: {stats['changes_15']} 回\n")
            f.write(f"30°以上: {stats['changes_30']} 回\n")
            f.write(f"60°以上: {stats['changes_60']} 回\n")

        print("\n生成された分析結果:")
        print(f"- 角度分析: {angle_path}")
        print(f"- 速度分析: {velocity_path}")
        print(f"- 経路分析: {path_path}")
        print(f"- 統計結果: {stats_path}")

        return {
            "angle_plot": angle_path,
            "velocity_plot": velocity_path,
            "path_plot": path_path,
            "stats": stats_path,
        }


def main():
    tracker = SimpleBugTracker()  # トラッカーインスタンス作成

    # 1. トラッキング実行
    csv_file = tracker.run_tracking()

    if csv_file:
        # 2. 分析実行
        print("\nデータ分析を開始します...")
        tracker.analyze_data(csv_file)
        print("\n=== 全ての分析が完了しました ===")
        print(f"結果ディレクトリ: {os.path.abspath(tracker.output_dir)}")


if __name__ == "__main__":
    main()
