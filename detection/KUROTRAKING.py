import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 初期設定
max_buffer = 100  # 軌跡ポイントのバッファサイズ
pts = deque(maxlen=max_buffer)  # 移動軌跡を保存するバッファ
speed_history = deque(maxlen=max_buffer)  # 速度記録
time_history = deque(maxlen=max_buffer)  # 時間記録
prev_pos = None  # 前回の位置
prev_time = time.time()  # 前回の時間

# HSVでの黒色範囲定義
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# ビデオキャプチャ初期化
cap = cv2.VideoCapture(0)  # 0はデフォルトカメラ

# グラフ設定
plt.ion()  # インタラクティブモードON
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
(line,) = ax1.plot([], [], "b-")  # 移動軌跡ライン
(speed_line,) = ax2.plot([], [], "r-")  # 速度ライン
ax1.set_title("Object Tracking Path")  # 軌跡グラフタイトル
ax2.set_title("Speed over Time")  # 速度グラフタイトル

# トラッキング制御変数 (True: 有効, False: 一時停止)
tracking_active = False
path_image = None  # 軌跡画像保存用

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 画面にトラッキング状態を表示
    status_text = f"Tracking: {'ACTIVE (Press p to pause)' if tracking_active else 'PAUSED (Press s to start)'}"
    cv2.putText(
        frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # トラッキングが有効な場合のみ処理
    if tracking_active:
        # 画像前処理
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV色空間に変換
        mask = cv2.inRange(hsv, lower_black, upper_black)  # 黒色領域をマスク
        mask = cv2.erode(mask, None, iterations=2)  # ノイズ除去（収縮）
        mask = cv2.dilate(mask, None, iterations=2)  # ノイズ除去（膨張）

        # 輪郭検出
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)  # 最大輪郭を選択
            ((x, y), radius) = cv2.minEnclosingCircle(c)  # 最小包囲円を取得
            M = cv2.moments(c)  # モーメント計算
            if M["m00"] > 0:
                center = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"]),
                )  # 重心計算
                current_time = time.time()
                pts.appendleft(center)  # 軌跡にポイント追加

                # 速度計算
                if prev_pos is not None:
                    time_diff = current_time - prev_time
                    if time_diff > 0:
                        distance = np.sqrt(
                            (center[0] - prev_pos[0]) ** 2
                            + (center[1] - prev_pos[1]) ** 2
                        )  # 移動距離計算
                        speed = distance / time_diff  # 速度計算
                        speed_history.append(speed)  # 速度記録
                        time_history.append(
                            current_time - time_history[0]
                            if len(time_history) > 0
                            else 0
                        )  # 相対時間記録

                prev_pos = center
                prev_time = current_time

        # 移動軌跡描画（保存用）
        if path_image is None:
            path_image = np.zeros_like(frame)  # 軌跡描画用黒画像初期化

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(max_buffer / float(i + 1)) * 2.5)  # 線の太さ調整
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)  # 画面描画
            cv2.line(
                path_image, pts[i - 1], pts[i], (0, 255, 0), thickness
            )  # 保存用描画

        # グラフ更新
        if len(pts) > 1 and len(speed_history) > 0:
            x_vals = [p[0] for p in pts if p is not None]  # X座標リスト
            y_vals = [p[1] for p in pts if p is not None]  # Y座標リスト
            line.set_data(x_vals, y_vals)  # 軌跡データ更新
            ax1.set_xlim(min(x_vals) - 50, max(x_vals) + 50)  # X軸範囲設定
            ax1.set_ylim(min(y_vals) - 50, max(y_vals) + 50)  # Y軸範囲設定

            speed_line.set_data(time_history, speed_history)  # 速度データ更新
            ax2.set_xlim(
                min(time_history),
                max(time_history) + 0.1 if len(time_history) > 0 else 1,
            )  # 時間軸範囲
            ax2.set_ylim(
                0, max(speed_history) + 10 if len(speed_history) > 0 else 100
            )  # 速度軸範囲

            plt.pause(0.001)  # グラフ更新

    # 画面表示
    cv2.imshow("Object Tracking", frame)

    # キー入力処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # トラッキング開始
        tracking_active = True
        print("Tracking STARTED")
    elif key == ord("p"):  # トラッキング一時停止
        tracking_active = False
        print("Tracking PAUSED")
    elif key == ord("q"):  # 終了＆保存
        if path_image is not None:
            combined_path = cv2.addWeighted(frame, 0.7, path_image, 0.3, 0)  # 画像合成
            cv2.imwrite("tracking_path.png", combined_path)
            print("Tracking path saved as 'tracking_path.png'")
        if len(speed_history) > 0:
            plt.savefig("speed_graph.png")  # グラフ保存
            print("Speed graph saved as 'speed_graph.png'")
        break

# 終了処理
cap.release()  # キャプチャ解放
cv2.destroyAllWindows()  # ウィンドウ閉じる
plt.ioff()  # インタラクティブモードOFF
plt.close()  # グラフ閉じる
