import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# 화면 크기 설정
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 사각형 크기 설정
RECT_SIZE = 0.5  # 사각형 크기 (0.5 x 0.5)
pattern = plt.Rectangle((5, 5), RECT_SIZE, RECT_SIZE, color="red")  # 중앙에서 시작
ax.add_patch(pattern)

# 속도 설정 (최대 속도 제한, 더 느리게 조정)
MAX_SPEED = 0.6  # 최대 속도
MIN_SPEED = 0.2  # 최소 속도
velocity = (np.random.rand(2) - 0.5) * (MAX_SPEED - MIN_SPEED) + np.sign(
    np.random.rand(2) - 0.5
) * MIN_SPEED


# 반사 각도를 랜덤하게 변환하는 함수
def random_bounce_angle(velocity, max_angle=15):
    angle = np.radians(
        np.random.uniform(-max_angle, max_angle)
    )  # -15도 ~ 15도 랜덤 변환
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # 회전 변환 행렬 적용 (벡터 회전)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    new_velocity = rotation_matrix @ velocity

    # 속도 크기를 유지하면서 방향만 변경
    return new_velocity / np.linalg.norm(new_velocity) * np.linalg.norm(velocity)


# 패턴이 경계에 닿았을 때 반사되도록 설정 (사각형 끝이 닿을 때)
def update(frame):
    global velocity

    # 패턴의 현재 위치
    x, y = pattern.get_xy()

    # 패턴의 새로운 위치 계산
    x += velocity[0]
    y += velocity[1]

    # 경계에서 사각형의 끝이 닿았을 때만 반사 (각도를 랜덤하게 변경)
    if x <= 0:  # 왼쪽 경계를 초과
        x = 0  # 경계 안으로 보정
        velocity[0] = -velocity[0]
        velocity = random_bounce_angle(velocity)  # 랜덤한 반사 적용
    elif x + RECT_SIZE >= 10:  # 오른쪽 경계를 초과
        x = 10 - RECT_SIZE  # 경계 안으로 보정
        velocity[0] = -velocity[0]
        velocity = random_bounce_angle(velocity)

    if y <= 0:  # 아래쪽 경계를 초과
        y = 0  # 경계 안으로 보정
        velocity[1] = -velocity[1]
        velocity = random_bounce_angle(velocity)
    elif y + RECT_SIZE >= 10:  # 위쪽 경계를 초과
        y = 10 - RECT_SIZE  # 경계 안으로 보정
        velocity[1] = -velocity[1]
        velocity = random_bounce_angle(velocity)

    # 패턴 위치 업데이트
    pattern.set_xy((x, y))

    return (pattern,)


# 애니메이션 생성 (interval을 줄여서 프레임을 부드럽게)
ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=True)

# 애니메이션 표시
plt.show()
