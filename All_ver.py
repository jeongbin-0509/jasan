import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 실험 환경 및 통제 변인 설정
# ==========================================
GRID_SIZE, VOXEL, MAX_DEPTH = 20, 10, 500
SCANNER_POS = (50, 50, 50)
TARGET_OBJ = {"x": 120, "y": 100, "z": 90, "r": 35}

def is_outside(ix, iy, iz):
    return (ix < 0 or ix >= GRID_SIZE or iy < 0 or iy >= GRID_SIZE or iz < 0 or iz >= GRID_SIZE)

def is_hit(x, y, z):
    return (x - TARGET_OBJ["x"])**2 + (y - TARGET_OBJ["y"])**2 + (z - TARGET_OBJ["z"])**2 <= TARGET_OBJ["r"]**2

def voxel_overlaps(ix, iy, iz):
    cx = max(ix*VOXEL, min(TARGET_OBJ["x"], (ix+1)*VOXEL))
    cy = max(iy*VOXEL, min(TARGET_OBJ["y"], (iy+1)*VOXEL))
    cz = max(iz*VOXEL, min(TARGET_OBJ["z"], (iz+1)*VOXEL))
    return (cx - TARGET_OBJ["x"])**2 + (cy - TARGET_OBJ["y"])**2 + (cz - TARGET_OBJ["z"])**2 <= TARGET_OBJ["r"]**2

def get_true_dist(px, py, pz, direction):
    # 오차율 계산의 기준이 되는 참값(Ground Truth) 추출용 함수
    dx, dy, dz = direction
    distance = 0
    while distance < MAX_DEPTH:
        x, y, z = px + dx * distance, py + dy * distance, pz + dz * distance
        if is_outside(int(x//VOXEL), int(y//VOXEL), int(z//VOXEL)): return MAX_DEPTH
        if is_hit(x, y, z): return distance
        distance += 0.05
    return MAX_DEPTH

# ==========================================
# 2. 알고리즘 1: 단순 Ray Casting
# ==========================================
def scan_ray_casting(px, py, pz, direction):
    dx, dy, dz = direction
    distance, checks = 0, 0
    while distance < MAX_DEPTH:
        checks += 1
        x, y, z = px + dx * distance, py + dy * distance, pz + dz * distance
        ix, iy, iz = int(x // VOXEL), int(y // VOXEL), int(z // VOXEL)
        if is_outside(ix, iy, iz): return MAX_DEPTH, checks
        if is_hit(x, y, z): return distance, checks
        distance += 0.05
    return MAX_DEPTH, checks

# ==========================================
# 3. 알고리즘 2: 기본 DDA
# ==========================================
def scan_dda(px, py, pz, direction):
    dx, dy, dz = direction
    x, y, z = px/VOXEL, py/VOXEL, pz/VOXEL
    ix, iy, iz = int(x), int(y), int(z)
    delX = abs(1/dx) if dx != 0 else float("inf")
    delY = abs(1/dy) if dy != 0 else float("inf")
    delZ = abs(1/dz) if dz != 0 else float("inf")
    stX, sdX = (-1, (x - ix) * delX) if dx < 0 else (1, (ix + 1 - x) * delX)
    stY, sdY = (-1, (y - iy) * delY) if dy < 0 else (1, (iy + 1 - y) * delY)
    stZ, sdZ = (-1, (z - iz) * delZ) if dz < 0 else (1, (iz + 1 - z) * delZ)

    checks, last_axis = 0, "x"
    while True:
        checks += 1
        if sdX < sdY and sdX < sdZ: sdX += delX; ix += stX; last_axis = "x"
        elif sdY < sdZ:             sdY += delY; iy += stY; last_axis = "y"
        else:                       sdZ += delZ; iz += stZ; last_axis = "z"

        if is_outside(ix, iy, iz): return MAX_DEPTH, checks
        if voxel_overlaps(ix, iy, iz): break

    if last_axis == "x":   dist = (ix - x + (1 - stX) / 2) / dx
    elif last_axis == "y": dist = (iy - y + (1 - stY) / 2) / dy
    else:                  dist = (iz - z + (1 - stZ) / 2) / dz
    return abs(dist * VOXEL), checks

# ==========================================
# 4. 알고리즘 3: Hybrid (제안 모델)
# ==========================================
def scan_hybrid(px, py, pz, direction):
    dx, dy, dz = direction
    x, y, z = px/VOXEL, py/VOXEL, pz/VOXEL
    ix, iy, iz = int(x), int(y), int(z)
    delX = abs(1/dx) if dx != 0 else float("inf")
    delY = abs(1/dy) if dy != 0 else float("inf")
    delZ = abs(1/dz) if dz != 0 else float("inf")
    stX, sdX = (-1, (x - ix) * delX) if dx < 0 else (1, (ix + 1 - x) * delX)
    stY, sdY = (-1, (y - iy) * delY) if dy < 0 else (1, (iy + 1 - y) * delY)
    stZ, sdZ = (-1, (z - iz) * delZ) if dz < 0 else (1, (iz + 1 - z) * delZ)

    checks, last_axis = 0, "x"
    
    # 1단계: DDA 고속 탐색
    while True:
        checks += 1
        if sdX < sdY and sdX < sdZ: sdX += delX; ix += stX; last_axis = "x"
        elif sdY < sdZ:             sdY += delY; iy += stY; last_axis = "y"
        else:                       sdZ += delZ; iz += stZ; last_axis = "z"

        if is_outside(ix, iy, iz): return MAX_DEPTH, checks
        if voxel_overlaps(ix, iy, iz): break

    if last_axis == "x":   approx_dist = (ix - x + (1 - stX) / 2) / dx
    elif last_axis == "y": approx_dist = (iy - y + (1 - stY) / 2) / dy
    else:                  approx_dist = (iz - z + (1 - stZ) / 2) / dz
    
    # 2단계: 에러 보정용 정밀 Ray Casting 전환
    fine_dist = max(0, abs(approx_dist * VOXEL) - (VOXEL / 2))
    while fine_dist < MAX_DEPTH:
        checks += 1
        rx, ry, rz = px + dx*fine_dist, py + dy*fine_dist, pz + dz*fine_dist
        if is_hit(rx, ry, rz): return fine_dist, checks
        if fine_dist > abs(approx_dist * VOXEL) + (VOXEL * 1.5): return MAX_DEPTH, checks
        fine_dist += 0.05
    return MAX_DEPTH, checks

# ==========================================
# 5. 실험 실행 및 시각화 저장
# ==========================================
if __name__ == "__main__":
    print("실험을 시작합니다. (데이터 도출 중...)")
    random.seed(42)
    ray_counts = [100, 300, 500, 1000, 2000]
    results = []

    algos = {
        "Ray Casting": scan_ray_casting,
        "DDA": scan_dda,
        "Hybrid": scan_hybrid
    }

    for count in ray_counts:
        # 광선(Ray) 방향 난수 생성
        directions = [(math.sin(p)*math.cos(t), math.sin(p)*math.sin(t), math.cos(p)) 
                      for t, p in [(random.uniform(0, 2*math.pi), random.uniform(0, math.pi)) for _ in range(count)]]
        
        # 참값 미리 계산
        true_dists = [get_true_dist(SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2], d) for d in directions]
        
        for name, algo in algos.items():
            start = time.perf_counter()
            tot_checks, tot_err = 0, 0
            
            for i, d in enumerate(directions):
                dist, checks = algo(SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2], d)
                tot_checks += checks
                
                # 빗나간 광선 처리 통일
                if dist != MAX_DEPTH and true_dists[i] != MAX_DEPTH:
                    tot_err += abs(dist - true_dists[i])
                elif dist != true_dists[i]:
                    tot_err += abs(dist - true_dists[i])
                    
            results.append({
                "Algorithm": name,
                "Ray Count": count,
                "Time(ms)": (time.perf_counter() - start) * 1000,
                "Avg Checks": tot_checks / count,
                "Avg Error": tot_err / count
            })

    # 1. 데이터 표(CSV) 저장
    df = pd.DataFrame(results)
    df.to_csv("3d_scan_benchmark_results.csv", index=False, encoding="utf-8-sig")

    # 2. 통합 시각화 그래프(PNG) 저장
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("3D Scan Algorithm Performance Benchmark", fontsize=16, fontweight='bold')
    
    colors = {"Ray Casting": "blue", "DDA": "orange", "Hybrid": "green"}
    
    for metric, ax, title in zip(["Time(ms)", "Avg Checks", "Avg Error"], axes, ["Processing Time (Lower is Better)", "Computation Checks (Lower is Better)", "Scan Accuracy Error (Lower is Better)"]):
        for algo in algos.keys():
            data = df[df["Algorithm"] == algo]
            ax.plot(data["Ray Count"], data[metric], marker='o', linewidth=2, label=algo, color=colors[algo])
        ax.set_title(title)
        ax.set_xlabel("Ray Count")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig("3d_scan_benchmark_graph.png", dpi=300)
    print("완료! 결과물이 저장되었습니다 (3d_scan_benchmark_results.csv, 3d_scan_benchmark_graph.png).")
