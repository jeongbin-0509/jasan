import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

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
    dx, dy, dz = direction
    distance = 0
    while distance < MAX_DEPTH:
        x, y, z = px + dx * distance, py + dy * distance, pz + dz * distance
        if is_outside(int(x//VOXEL), int(y//VOXEL), int(z//VOXEL)): return MAX_DEPTH
        if is_hit(x, y, z): return distance
        distance += 0.05
    return MAX_DEPTH

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

if __name__ == "__main__":
    print("[2] DDA 단독 실험 시작...")
    random.seed(42)
    ray_counts = [100, 300, 500, 1000, 2000]
    results = []

    for count in ray_counts:
        directions = [(math.sin(p)*math.cos(t), math.sin(p)*math.sin(t), math.cos(p)) 
                      for t, p in [(random.uniform(0, 2*math.pi), random.uniform(0, math.pi)) for _ in range(count)]]
        
        true_dists = [get_true_dist(SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2], d) for d in directions]
        
        start = time.perf_counter()
        tot_checks, tot_err = 0, 0
        
        for i, d in enumerate(directions):
            dist, checks = scan_dda(SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2], d)
            tot_checks += checks
            if dist != MAX_DEPTH and true_dists[i] != MAX_DEPTH:
                tot_err += abs(dist - true_dists[i])
            elif dist != true_dists[i]:
                tot_err += abs(dist - true_dists[i])
                
        results.append({
            "Ray Count": count,
            "Time(ms)": (time.perf_counter() - start) * 1000,
            "Avg Checks": tot_checks / count,
            "Avg Error": tot_err / count
        })

    df = pd.DataFrame(results)
    df.to_csv("2_dda_results.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("DDA Limitations (Fast but High Error)", fontsize=14, fontweight='bold')
    
    axes[0].plot(df["Ray Count"], df["Time(ms)"], marker='o', color='orange')
    axes[0].set_title("Time(ms) - Very Fast")
    
    axes[1].plot(df["Ray Count"], df["Avg Checks"], marker='o', color='orange')
    axes[1].set_title("Avg Checks - Very Low")
    
    axes[2].plot(df["Ray Count"], df["Avg Error"], marker='o', color='orange')
    axes[2].set_title("Avg Error - High Error (Blocky)")

    plt.tight_layout()
    plt.savefig("2_dda_graph.png", dpi=300)
    print("완료! 결과 저장됨: 2_dda_results.csv, 2_dda_graph.png")
