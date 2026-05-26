import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

# 3D 공간 설명

GRID_SIZE = 20
VOXEL = 10
MAX_DEPTH = 500

PLAYER = (50, 50, 50)

# 구 장애물
SPHERE = {
    "x": 120,
    "y": 100,
    "z": 90,
    "r": 35
}

# 기본 함수

def is_outside(ix, iy, iz):
    return (
        ix < 0 or ix >= GRID_SIZE or
        iy < 0 or iy >= GRID_SIZE or
        iz < 0 or iz >= GRID_SIZE
    )


def is_sphere_hit(x, y, z):
    dx = x - SPHERE["x"]
    dy = y - SPHERE["y"]
    dz = z - SPHERE["z"]

    return dx * dx + dy * dy + dz * dz <= SPHERE["r"] ** 2


def voxel_overlaps_sphere(ix, iy, iz):
    left = ix * VOXEL
    right = left + VOXEL

    bottom = iy * VOXEL
    top = bottom + VOXEL

    back = iz * VOXEL
    front = back + VOXEL

    closest_x = max(left, min(SPHERE["x"], right))
    closest_y = max(bottom, min(SPHERE["y"], top))
    closest_z = max(back, min(SPHERE["z"], front))

    dx = SPHERE["x"] - closest_x
    dy = SPHERE["y"] - closest_y
    dz = SPHERE["z"] - closest_z

    return dx * dx + dy * dy + dz * dz <= SPHERE["r"] ** 2


# 1. 3D Ray Casting

def ray_casting_3d(px, py, pz, direction, step=1):
    dx, dy, dz = direction

    distance = 0
    checks = 0

    while distance < MAX_DEPTH:
        checks += 1

        x = px + dx * distance
        y = py + dy * distance
        z = pz + dz * distance

        ix = int(x // VOXEL)
        iy = int(y // VOXEL)
        iz = int(z // VOXEL)

        if is_outside(ix, iy, iz):
            return distance, checks

        if is_sphere_hit(x, y, z):
            return distance, checks

        distance += step

    return MAX_DEPTH, checks


# 2. 3D DDA

def dda_3d(px, py, pz, direction):
    dx, dy, dz = direction

    x = px / VOXEL
    y = py / VOXEL
    z = pz / VOXEL

    ix = int(x)
    iy = int(y)
    iz = int(z)

    delta_x = abs(1 / dx) if dx != 0 else float("inf")
    delta_y = abs(1 / dy) if dy != 0 else float("inf")
    delta_z = abs(1 / dz) if dz != 0 else float("inf")

    if dx < 0:
        step_x = -1
        side_x = (x - ix) * delta_x
    else:
        step_x = 1
        side_x = (ix + 1 - x) * delta_x

    if dy < 0:
        step_y = -1
        side_y = (y - iy) * delta_y
    else:
        step_y = 1
        side_y = (iy + 1 - y) * delta_y

    if dz < 0:
        step_z = -1
        side_z = (z - iz) * delta_z
    else:
        step_z = 1
        side_z = (iz + 1 - z) * delta_z

    checks = 0
    last_axis = "x"

    while True:
        checks += 1

        if side_x < side_y and side_x < side_z:
            side_x += delta_x
            ix += step_x
            last_axis = "x"

        elif side_y < side_z:
            side_y += delta_y
            iy += step_y
            last_axis = "y"

        else:
            side_z += delta_z
            iz += step_z
            last_axis = "z"

        if is_outside(ix, iy, iz):
            return MAX_DEPTH, checks

        if voxel_overlaps_sphere(ix, iy, iz):
            break

    if last_axis == "x":
        distance = (ix - x + (1 - step_x) / 2) / dx
    elif last_axis == "y":
        distance = (iy - y + (1 - step_y) / 2) / dy
    else:
        distance = (iz - z + (1 - step_z) / 2) / dz

    return abs(distance * VOXEL), checks


# 랜덤 3D 방향 생성

def random_direction():
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)

    dx = math.sin(phi) * math.cos(theta)
    dy = math.sin(phi) * math.sin(theta)
    dz = math.cos(phi)

    return dx, dy, dz


# 실험

def experiment(ray_counts):
    results = []

    for ray_count in ray_counts:
        directions = [random_direction() for _ in range(ray_count)]

        true_distances = []

        for direction in directions:
            dist, _ = ray_casting_3d(
                PLAYER[0],
                PLAYER[1],
                PLAYER[2],
                direction,
                step=0.2
            )

            true_distances.append(dist)

        algorithms = {
            "3D Ray Casting": ray_casting_3d,
            "3D DDA": dda_3d
        }

        for name, algorithm in algorithms.items():
            start = time.perf_counter()

            total_checks = 0
            total_error = 0

            for i, direction in enumerate(directions):
                dist, checks = algorithm(
                    PLAYER[0],
                    PLAYER[1],
                    PLAYER[2],
                    direction
                )

                total_checks += checks
                total_error += abs(dist - true_distances[i])

            end = time.perf_counter()

            results.append({
                "Algorithm": name,
                "Ray Count": ray_count,
                "Time(ms)": (end - start) * 1000,
                "Avg Checks": total_checks / ray_count,
                "Avg Error": total_error / ray_count
            })

    return pd.DataFrame(results)


# 그래프 저장

def draw_graph(df):
    metrics = [
        "Time(ms)",
        "Avg Checks",
        "Avg Error"
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for algorithm in df["Algorithm"].unique():
            data = df[df["Algorithm"] == algorithm]

            plt.plot(
                data["Ray Count"],
                data[metric],
                marker="o",
                label=algorithm
            )

        plt.title(metric)
        plt.xlabel("Ray Count")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = metric.replace("(", "_").replace(")", "").replace("/", "_") + "_3d.png"

        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"그래프 저장 완료: {filename}")


# 실행

if __name__ == "__main__":
    random.seed(42)

    ray_counts = [100, 300, 500, 1000, 2000]

    df = experiment(ray_counts)

    print(df)

    df.to_csv(
        "raycasting_vs_dda_3d_sphere.csv",
        index=False,
        encoding="utf-8-sig"
    )

    draw_graph(df)

    print("실험 완료")
    print("CSV 저장: raycasting_vs_dda_3d_sphere.csv")