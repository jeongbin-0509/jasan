import math
import time
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 기본 설정
# =========================
MAX_DEPTH = 300
VOXEL = 5
GRID_SIZE = 80
WORLD_SIZE = GRID_SIZE * VOXEL

CAR_POS = (50, 200, 20)   # 자율주행차 센서 위치
RAY_COUNT_LIST = [100, 300, 500, 1000, 2000]
TARGET_SPEED_LIST = [1, 3, 5, 8]  # 움직이는 대상 속도
FRAMES = 60


# =========================
# 움직이는 대상 설정
# =========================
def get_moving_target(frame, speed):
    """
    움직이는 보행자/차량을 박스 형태로 단순화
    x 방향으로 이동하는 동적 장애물
    """
    x = 160 + frame * speed
    y = 200
    z = 0

    return {
        "name": "moving_object",
        "min": (x - 10, y - 10, z),
        "max": (x + 10, y + 10, z + 40)
    }


# =========================
# 충돌 판정 함수
# =========================
def is_outside(x, y, z):
    return x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE or z < 0 or z >= WORLD_SIZE


def point_in_box(x, y, z, obj):
    min_x, min_y, min_z = obj["min"]
    max_x, max_y, max_z = obj["max"]

    return (
        min_x <= x <= max_x and
        min_y <= y <= max_y and
        min_z <= z <= max_z
    )


def is_hit(x, y, z, obj):
    return point_in_box(x, y, z, obj)


def voxel_overlaps_box(ix, iy, iz, obj):
    vx_min, vy_min, vz_min = ix * VOXEL, iy * VOXEL, iz * VOXEL
    vx_max, vy_max, vz_max = (ix + 1) * VOXEL, (iy + 1) * VOXEL, (iz + 1) * VOXEL

    ox_min, oy_min, oz_min = obj["min"]
    ox_max, oy_max, oz_max = obj["max"]

    return (
        vx_min <= ox_max and vx_max >= ox_min and
        vy_min <= oy_max and vy_max >= oy_min and
        vz_min <= oz_max and vz_max >= oz_min
    )


# =========================
# 기준 거리 계산
# =========================
def get_true_dist(px, py, pz, direction, obj):
    dx, dy, dz = direction
    dist = 0

    while dist < MAX_DEPTH:
        x = px + dx * dist
        y = py + dy * dist
        z = pz + dz * dist

        if is_outside(x, y, z):
            return MAX_DEPTH

        if is_hit(x, y, z, obj):
            return dist

        dist += 0.05

    return MAX_DEPTH


# =========================
# Hybrid 알고리즘
# =========================
def scan_hybrid(px, py, pz, direction, obj):
    dx, dy, dz = direction

    x, y, z = px / VOXEL, py / VOXEL, pz / VOXEL
    ix, iy, iz = int(x), int(y), int(z)

    del_x = abs(1 / dx) if dx != 0 else float("inf")
    del_y = abs(1 / dy) if dy != 0 else float("inf")
    del_z = abs(1 / dz) if dz != 0 else float("inf")

    step_x, side_x = (-1, (x - ix) * del_x) if dx < 0 else (1, (ix + 1 - x) * del_x)
    step_y, side_y = (-1, (y - iy) * del_y) if dy < 0 else (1, (iy + 1 - y) * del_y)
    step_z, side_z = (-1, (z - iz) * del_z) if dz < 0 else (1, (iz + 1 - z) * del_z)

    checks = 0
    last_axis = "x"

    # 1차: DDA로 빠르게 대상 근처까지 탐색
    while True:
        checks += 1

        if side_x < side_y and side_x < side_z:
            side_x += del_x
            ix += step_x
            last_axis = "x"
        elif side_y < side_z:
            side_y += del_y
            iy += step_y
            last_axis = "y"
        else:
            side_z += del_z
            iz += step_z
            last_axis = "z"

        if ix < 0 or ix >= GRID_SIZE or iy < 0 or iy >= GRID_SIZE or iz < 0 or iz >= GRID_SIZE:
            return MAX_DEPTH, checks

        if voxel_overlaps_box(ix, iy, iz, obj):
            break

    # DDA로 구한 대략 거리
    if last_axis == "x":
        approx_dist = (ix - x + (1 - step_x) / 2) / dx
    elif last_axis == "y":
        approx_dist = (iy - y + (1 - step_y) / 2) / dy
    else:
        approx_dist = (iz - z + (1 - step_z) / 2) / dz

    approx_dist = abs(approx_dist * VOXEL)

    # 2차: 근처에서만 Ray Casting으로 정밀 보정
    fine_dist = max(0, approx_dist - VOXEL)

    while fine_dist < approx_dist + VOXEL * 2:
        checks += 1

        rx = px + dx * fine_dist
        ry = py + dy * fine_dist
        rz = pz + dz * fine_dist

        if is_outside(rx, ry, rz):
            return MAX_DEPTH, checks

        if is_hit(rx, ry, rz, obj):
            return fine_dist, checks

        fine_dist += 0.05

    return MAX_DEPTH, checks


# =========================
# 자율주행차 전방 방향 광선 생성
# =========================
def generate_front_rays(ray_count):
    directions = []

    # 차량이 +x 방향을 바라본다고 가정
    # 좌우 시야각 90도, 상하 시야각 약간 포함
    for i in range(ray_count):
        h_angle = -math.pi / 4 + (math.pi / 2) * (i / max(1, ray_count - 1))
        v_angle = 0

        dx = math.cos(h_angle)
        dy = math.sin(h_angle)
        dz = math.sin(v_angle)

        directions.append((dx, dy, dz))

    return directions


# =========================
# 실험 실행
# =========================
def run_autonomous_vehicle_experiment():
    results = []

    for speed in TARGET_SPEED_LIST:
        for ray_count in RAY_COUNT_LIST:
            directions = generate_front_rays(ray_count)

            total_time = 0
            total_checks = 0
            total_error = 0
            total_detected = 0
            total_frames = 0

            for frame in range(FRAMES):
                target = get_moving_target(frame, speed)

                true_dists = [
                    get_true_dist(CAR_POS[0], CAR_POS[1], CAR_POS[2], d, target)
                    for d in directions
                ]

                start = time.perf_counter()

                frame_detected = False

                for i, d in enumerate(directions):
                    dist, checks = scan_hybrid(CAR_POS[0], CAR_POS[1], CAR_POS[2], d, target)

                    total_checks += checks

                    true_dist = true_dists[i]

                    if dist != MAX_DEPTH:
                        frame_detected = True

                    if dist != MAX_DEPTH and true_dist != MAX_DEPTH:
                        total_error += abs(dist - true_dist)
                    elif dist != true_dist:
                        total_error += abs(dist - true_dist)

                total_time += (time.perf_counter() - start) * 1000

                if frame_detected:
                    total_detected += 1

                total_frames += 1

            results.append({
                "Target Speed": speed,
                "Ray Count": ray_count,
                "Avg Time per Frame(ms)": total_time / FRAMES,
                "Avg Checks per Ray": total_checks / (FRAMES * ray_count),
                "Avg Error": total_error / (FRAMES * ray_count),
                "Detection Success Rate(%)": total_detected / total_frames * 100
            })

            print(
                f"Speed={speed}, Rays={ray_count} | "
                f"Time={total_time / FRAMES:.2f}ms | "
                f"Detect={total_detected / total_frames * 100:.1f}%"
            )

    df = pd.DataFrame(results)
    df.to_csv("autonomous_vehicle_hybrid_experiment.csv", index=False, encoding="utf-8-sig")
    return df


# =========================
# 그래프 저장
# =========================
def save_graphs(df):
    metrics = [
        "Avg Time per Frame(ms)",
        "Avg Checks per Ray",
        "Avg Error",
        "Detection Success Rate(%)"
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for speed in df["Target Speed"].unique():
            sub = df[df["Target Speed"] == speed]
            plt.plot(sub["Ray Count"], sub[metric], marker="o", label=f"Speed {speed}")

        plt.title(f"Hybrid Autonomous Vehicle - {metric}")
        plt.xlabel("Ray Count")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"auto_vehicle_{metric}"
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "percent")
        )

        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    df = run_autonomous_vehicle_experiment()
    save_graphs(df)

    print("\n자율주행 자동차 Hybrid 실험 완료!")
    print("저장 파일: autonomous_vehicle_hybrid_experiment.csv")