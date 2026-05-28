import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 3D 공간 및 실험 환경 통제
# =========================
GRID_SIZE = 20
VOXEL = 10
MAX_DEPTH = 500

PLAYER = (50, 50, 50)
SPHERE = {"x": 120, "y": 100, "z": 90, "r": 35}

def is_outside(ix, iy, iz):
    return (ix < 0 or ix >= GRID_SIZE or iy < 0 or iy >= GRID_SIZE or iz < 0 or iz >= GRID_SIZE)

def is_sphere_hit(x, y, z):
    dx, dy, dz = x - SPHERE["x"], y - SPHERE["y"], z - SPHERE["z"]
    return dx**2 + dy**2 + dz**2 <= SPHERE["r"]**2

def voxel_overlaps_sphere(ix, iy, iz):
    left, right = ix * VOXEL, (ix + 1) * VOXEL
    bottom, top = iy * VOXEL, (iy + 1) * VOXEL
    back, front = iz * VOXEL, (iz + 1) * VOXEL

    cx = max(left, min(SPHERE["x"], right))
    cy = max(bottom, min(SPHERE["y"], top))
    cz = max(back, min(SPHERE["z"], front))

    return (SPHERE["x"] - cx)**2 + (SPHERE["y"] - cy)**2 + (SPHERE["z"] - cz)**2 <= SPHERE["r"]**2

# =========================
# 2. 알고리즘 1: 단순 Ray Casting
# =========================
def ray_casting_3d(px, py, pz, direction, step=1):
    dx, dy, dz = direction
    distance = 0
    checks = 0

    while distance < MAX_DEPTH:
        checks += 1
        x, y, z = px + dx * distance, py + dy * distance, pz + dz * distance
        ix, iy, iz = int(x // VOXEL), int(y // VOXEL), int(z // VOXEL)

        if is_outside(ix, iy, iz) or is_sphere_hit(x, y, z):
            return distance, checks
        distance += step

    return MAX_DEPTH, checks

# =========================
# 3. 알고리즘 2: 기본 DDA
# =========================
def dda_3d(px, py, pz, direction):
    dx, dy, dz = direction
    x, y, z = px / VOXEL, py / VOXEL, pz / VOXEL
    ix, iy, iz = int(x), int(y), int(z)

    delta_x = abs(1 / dx) if dx != 0 else float("inf")
    delta_y = abs(1 / dy) if dy != 0 else float("inf")
    delta_z = abs(1 / dz) if dz != 0 else float("inf")

    step_x, side_x = (-1, (x - ix) * delta_x) if dx < 0 else (1, (ix + 1 - x) * delta_x)
    step_y, side_y = (-1, (y - iy) * delta_y) if dy < 0 else (1, (iy + 1 - y) * delta_y)
    step_z, side_z = (-1, (z - iz) * delta_z) if dz < 0 else (1, (iz + 1 - z) * delta_z)

    checks = 0
    last_axis = "x"

    while True:
        checks += 1
        if side_x < side_y and side_x < side_z:
            side_x += delta_x; ix += step_x; last_axis = "x"
        elif side_y < side_z:
            side_y += delta_y; iy += step_y; last_axis = "y"
        else:
            side_z += delta_z; iz += step_z; last_axis = "z"

        if is_outside(ix, iy, iz): return MAX_DEPTH, checks
        if voxel_overlaps_sphere(ix, iy, iz): break

    if last_axis == "x":   distance = (ix - x + (1 - step_x) / 2) / dx
    elif last_axis == "y": distance = (iy - y + (1 - step_y) / 2) / dy
    else:                  distance = (iz - z + (1 - step_z) / 2) / dz

    return abs(distance * VOXEL), checks

# =========================
# 4. 알고리즘 3: DDA + Ray 하이브리드 (우리 탐구의 핵심 융합 모델)
# =========================
def hybrid_dda_ray_3d(px, py, pz, direction):
    dx, dy, dz = direction
    x, y, z = px / VOXEL, py / VOXEL, pz / VOXEL
    ix, iy, iz = int(x), int(y), int(z)

    delta_x = abs(1 / dx) if dx != 0 else float("inf")
    delta_y = abs(1 / dy) if dy != 0 else float("inf")
    delta_z = abs(1 / dz) if dz != 0 else float("inf")

    step_x, side_x = (-1, (x - ix) * delta_x) if dx < 0 else (1, (ix + 1 - x) * delta_x)
    step_y, side_y = (-1, (y - iy) * delta_y) if dy < 0 else (1, (iy + 1 - y) * delta_y)
    step_z, side_z = (-1, (z - iz) * delta_z) if dz < 0 else (1, (iz + 1 - z) * delta_z)

    checks = 0
    last_axis = "x"
    
    # 1단계: DDA 고속 탐색 (빈 공간 건너뛰기)
    while True:
        checks += 1
        if side_x < side_y and side_x < side_z:
            side_x += delta_x; ix += step_x; last_axis = "x"
        elif side_y < side_z:
            side_y += delta_y; iy += step_y; last_axis = "y"
        else:
            side_z += delta_z; iz += step_z; last_axis = "z"

        if is_outside(ix, iy, iz): return MAX_DEPTH, checks
        if voxel_overlaps_sphere(ix, iy, iz): break

    # 2단계: 정밀 Ray Casting 전환 (장애물 표면 정밀 타격)
    if last_axis == "x":   approx_dist = (ix - x + (1 - step_x) / 2) / dx
    elif last_axis == "y": approx_dist = (iy - y + (1 - step_y) / 2) / dy
    else:                  approx_dist = (iz - z + (1 - step_z) / 2) / dz
    
    fine_dist = max(0, abs(approx_dist * VOXEL) - VOXEL)
    
    while fine_dist < MAX_DEPTH:
        checks += 1
        rx, ry, rz = px + dx * fine_dist, py + dy * fine_dist, pz + dz * fine_dist
        
        if int(rx // VOXEL) != ix and int(ry // VOXEL) != iy and int(rz // VOXEL) != iz:
            if not is_sphere_hit(rx, ry, rz): break

        if is_sphere_hit(rx, ry, rz): return fine_dist, checks
        fine_dist += 0.2

    return MAX_DEPTH, checks

# =========================
# 5. 실험 실행 및 3파전 비교 그래프 도출
# =========================
def experiment(ray_counts):
    results = []
    for count in ray_counts:
        directions = [(math.sin(p)*math.cos(t), math.sin(p)*math.sin(t), math.cos(p)) 
                      for t, p in [(random.uniform(0, 2*math.pi), random.uniform(0, math.pi)) for _ in range(count)]]
        
        true_distances = [ray_casting_3d(PLAYER[0], PLAYER[1], PLAYER[2], d, step=0.05)[0] for d in directions]

        algorithms = {
            "Ray Casting (Step=1)": lambda px, py, pz, d: ray_casting_3d(px, py, pz, d, step=1),
            "DDA (Voxel)": dda_3d,
            "Hybrid (Proposed)": hybrid_dda_ray_3d
        }

        for name, algo in algorithms.items():
            start = time.perf_counter()
            tot_checks, tot_err = 0, 0

            for i, dir_vec in enumerate(directions):
                dist, checks = algo(PLAYER[0], PLAYER[1], PLAYER[2], dir_vec)
                tot_checks += checks
                tot_err += abs(dist - true_distances[i])

            results.append({
                "Algorithm": name, "Ray Count": count,
                "Time(ms)": (time.perf_counter() - start) * 1000,
                "Avg Checks": tot_checks / count, "Avg Error": tot_err / count
            })
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("🚀 3개 알고리즘 비교 벤치마크를 시작합니다. (약 5~10초 소요)")
    random.seed(42)
    df = experiment([100, 300, 500, 1000, 2000])
    
    # 표 저장
    df.to_csv("hybrid_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 데이터 표 저장 완료: hybrid_results.csv")
    
    # 그래프 3개 저장
    for metric in ["Time(ms)", "Avg Checks", "Avg Error"]:
        plt.figure(figsize=(8, 5))
        for algo in df["Algorithm"].unique():
            data = df[df["Algorithm"] == algo]
            plt.plot(data["Ray Count"], data[metric], marker="o", linewidth=2, label=algo)
        
        plt.title(f"Performance Comparison: {metric}", fontsize=14, fontweight='bold')
        plt.xlabel("Ray Count (Number of Rays)", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        filename = f"graph_{metric.replace('(ms)','').strip()}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ 그래프 저장 완료: {filename}")
        
    print("\n🎉 모든 실험이 끝났습니다! 저장된 3장의 그래프를 보고서에 붙여넣으세요.")
