import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

MAX_DEPTH = 500
SCANNER_POS = (50, 50, 50)
random.seed(42)

OBJECTS = [
    # 외벽
    {"name": "front_wall", "min": (0, 0, 0), "max": (200, 10, 120)},
    {"name": "back_wall", "min": (0, 190, 0), "max": (200, 200, 120)},
    {"name": "left_wall", "min": (0, 0, 0), "max": (10, 200, 120)},
    {"name": "right_wall", "min": (190, 0, 0), "max": (200, 200, 120)},

    # 내부 벽
    {"name": "room_wall_1", "min": (70, 20, 0), "max": (80, 150, 100)},
    {"name": "room_wall_2", "min": (120, 60, 0), "max": (130, 190, 100)},

    # 기둥
    {"name": "pillar_1", "min": (100, 100, 0), "max": (115, 115, 120)},
    {"name": "pillar_2", "min": (150, 45, 0), "max": (165, 60, 120)},

    # 장애물
    {"name": "desk", "min": (40, 130, 0), "max": (65, 160, 45)},
    {"name": "cabinet", "min": (140, 130, 0), "max": (180, 155, 80)}
]


def is_outside(x, y, z, world_size):
    return x < 0 or x >= world_size or y < 0 or y >= world_size or z < 0 or z >= world_size


def point_in_box(x, y, z, obj):
    min_x, min_y, min_z = obj["min"]
    max_x, max_y, max_z = obj["max"]

    return (
        min_x <= x <= max_x and
        min_y <= y <= max_y and
        min_z <= z <= max_z
    )


def is_hit(x, y, z, objects):
    for obj in objects:
        if point_in_box(x, y, z, obj):
            return True
    return False


def voxel_overlaps_box(ix, iy, iz, voxel, obj):
    vx_min, vy_min, vz_min = ix * voxel, iy * voxel, iz * voxel
    vx_max, vy_max, vz_max = (ix + 1) * voxel, (iy + 1) * voxel, (iz + 1) * voxel

    ox_min, oy_min, oz_min = obj["min"]
    ox_max, oy_max, oz_max = obj["max"]

    return (
        vx_min <= ox_max and vx_max >= ox_min and
        vy_min <= oy_max and vy_max >= oy_min and
        vz_min <= oz_max and vz_max >= oz_min
    )


def voxel_overlaps(ix, iy, iz, voxel, objects):
    for obj in objects:
        if voxel_overlaps_box(ix, iy, iz, voxel, obj):
            return True
    return False


def get_true_dist(px, py, pz, direction, world_size, objects):
    dx, dy, dz = direction
    dist = 0

    while dist < MAX_DEPTH:
        x = px + dx * dist
        y = py + dy * dist
        z = pz + dz * dist

        if is_outside(x, y, z, world_size):
            return MAX_DEPTH

        if is_hit(x, y, z, objects):
            return dist

        dist += 0.05

    return MAX_DEPTH


def scan_hybrid(px, py, pz, direction, grid_size, voxel, world_size, objects):
    dx, dy, dz = direction

    x, y, z = px / voxel, py / voxel, pz / voxel
    ix, iy, iz = int(x), int(y), int(z)

    del_x = abs(1 / dx) if dx != 0 else float("inf")
    del_y = abs(1 / dy) if dy != 0 else float("inf")
    del_z = abs(1 / dz) if dz != 0 else float("inf")

    step_x, side_x = (-1, (x - ix) * del_x) if dx < 0 else (1, (ix + 1 - x) * del_x)
    step_y, side_y = (-1, (y - iy) * del_y) if dy < 0 else (1, (iy + 1 - y) * del_y)
    step_z, side_z = (-1, (z - iz) * del_z) if dz < 0 else (1, (iz + 1 - z) * del_z)

    checks = 0
    last_axis = "x"

    # 1차: DDA로 목표물 근처까지 빠르게 접근
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

        if ix < 0 or ix >= grid_size or iy < 0 or iy >= grid_size or iz < 0 or iz >= grid_size:
            return MAX_DEPTH, checks

        if voxel_overlaps(ix, iy, iz, voxel, objects):
            break

    # DDA로 얻은 대략적인 충돌 거리
    if last_axis == "x":
        approx_dist = (ix - x + (1 - step_x) / 2) / dx
    elif last_axis == "y":
        approx_dist = (iy - y + (1 - step_y) / 2) / dy
    else:
        approx_dist = (iz - z + (1 - step_z) / 2) / dz

    approx_dist = abs(approx_dist * voxel)

    # 2차: 목표물 근처에서만 Ray Casting 정밀 보정
    fine_dist = max(0, approx_dist - voxel)

    while fine_dist < approx_dist + voxel * 2:
        checks += 1

        rx = px + dx * fine_dist
        ry = py + dy * fine_dist
        rz = pz + dz * fine_dist

        if is_outside(rx, ry, rz, world_size):
            return MAX_DEPTH, checks

        if is_hit(rx, ry, rz, objects):
            return fine_dist, checks

        fine_dist += 0.05

    return MAX_DEPTH, checks


def generate_directions(count):
    directions = []

    for _ in range(count):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)

        dx = math.sin(phi) * math.cos(theta)
        dy = math.sin(phi) * math.sin(theta)
        dz = math.cos(phi)

        directions.append((dx, dy, dz))

    return directions


def run_hybrid_real_life_experiment():
    scenarios = [
        {
            "Scenario": "Basic Indoor Space",
            "GRID_SIZE": 20,
            "VOXEL": 10,
            "Ray Counts": [100, 300, 500, 1000, 2000]
        },
        {
            "Scenario": "High Load Space",
            "GRID_SIZE": 20,
            "VOXEL": 10,
            "Ray Counts": [1000, 3000, 5000, 10000]
        },
        {
            "Scenario": "Low Memory Space",
            "GRID_SIZE": 10,
            "VOXEL": 20,
            "Ray Counts": [100, 300, 500, 1000, 2000]
        },
        {
            "Scenario": "High Precision Space",
            "GRID_SIZE": 40,
            "VOXEL": 5,
            "Ray Counts": [100, 300, 500, 1000, 2000]
        },
        {
            "Scenario": "Extreme Space",
            "GRID_SIZE": 40,
            "VOXEL": 5,
            "Ray Counts": [1000, 3000, 5000, 10000]
        }
    ]

    results = []

    for scenario in scenarios:
        scenario_name = scenario["Scenario"]
        grid_size = scenario["GRID_SIZE"]
        voxel = scenario["VOXEL"]
        world_size = grid_size * voxel

        for ray_count in scenario["Ray Counts"]:
            directions = generate_directions(ray_count)

            true_dists = [
                get_true_dist(
                    SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2],
                    d, world_size, OBJECTS
                )
                for d in directions
            ]

            start = time.perf_counter()

            total_checks = 0
            total_error = 0
            hit_success = 0

            for i, d in enumerate(directions):
                dist, checks = scan_hybrid(
                    SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2],
                    d, grid_size, voxel, world_size, OBJECTS
                )

                total_checks += checks
                true_dist = true_dists[i]

                if dist != MAX_DEPTH and true_dist != MAX_DEPTH:
                    hit_success += 1
                    total_error += abs(dist - true_dist)
                elif dist != true_dist:
                    total_error += abs(dist - true_dist)

            elapsed_ms = (time.perf_counter() - start) * 1000

            results.append({
                "Scenario": scenario_name,
                "GRID_SIZE": grid_size,
                "VOXEL": voxel,
                "Ray Count": ray_count,
                "World Size": world_size,
                "Time(ms)": elapsed_ms,
                "Avg Checks": total_checks / ray_count,
                "Avg Error": total_error / ray_count,
                "Hit Success Rate(%)": hit_success / ray_count * 100
            })

            print(
                f"{scenario_name} | Rays={ray_count} "
                f"| Time={elapsed_ms:.2f}ms "
                f"| Avg Checks={total_checks / ray_count:.2f} "
                f"| Avg Error={total_error / ray_count:.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv("hybrid_real_life_experiment.csv", index=False, encoding="utf-8-sig")
    return df


def save_hybrid_graphs(df):
    metrics = ["Time(ms)", "Avg Checks", "Avg Error", "Hit Success Rate(%)"]

    for metric in metrics:
        plt.figure(figsize=(9, 5))

        for scenario in df["Scenario"].unique():
            sub = df[df["Scenario"] == scenario]

            plt.plot(
                sub["Ray Count"],
                sub[metric],
                marker="o",
                label=scenario
            )

        plt.title(f"Hybrid Algorithm - {metric}")
        plt.xlabel("Ray Count")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"hybrid_{metric}"
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "percent")
        )

        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    df = run_hybrid_real_life_experiment()
    save_hybrid_graphs(df)

    print("\nHybrid 실생활 환경 적용 실험 완료!")
    print("저장 파일: hybrid_real_life_experiment.csv")
    print("그래프 PNG 파일들이 함께 저장되었습니다.")