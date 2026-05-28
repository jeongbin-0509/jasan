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

if __name__ == "__main__":
    print("[1] Ray Casting 단독 실험 시작...")
    random.seed(42)
    ray_counts = [100, 300, 500, 1000, 2000]
    results = []

    for count in ray_counts:
        directions = [(math.sin(p)*math.cos(t), math.sin(p)*math.sin(t), math.cos(p)) 
                      for t, p in [(random.uniform(0, 2*math.pi), random.uniform(0, math.pi)) for _ in range(count)]]
        
        start = time.perf_counter()
        tot_checks, tot_err = 0, 0
        
        for d in directions:
            dist, checks = scan_ray_casting(SCANNER_POS[0], SCANNER_POS[1], SCANNER_POS[2], d)
            tot_checks += checks
            # Ray Casting 자체가 참값이므로 오차는 0
            
        results.append({
            "Ray Count": count,
            "Time(ms)": (time.perf_counter() - start) * 1000,
            "Avg Checks": tot_checks / count,
            "Avg Error": 0.0
        })

    df = pd.DataFrame(results)
    df.to_csv("1_ray_casting_results.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Ray Casting Limitations (Accurate but Slow)", fontsize=14, fontweight='bold')
    
    axes[0].plot(df["Ray Count"], df["Time(ms)"], marker='o', color='blue')
    axes[0].set_title("Time(ms) - Exploding")
    
    axes[1].plot(df["Ray Count"], df["Avg Checks"], marker='o', color='blue')
    axes[1].set_title("Avg Checks - Very High")
    
    axes[2].plot(df["Ray Count"], df["Avg Error"], marker='o', color='blue')
    axes[2].set_title("Avg Error - Perfect (0)")

    plt.tight_layout()
    plt.savefig("1_ray_casting_graph.png", dpi=300)
    print("완료! 결과 저장됨: 1_ray_casting_results.csv, 1_ray_casting_graph.png")
