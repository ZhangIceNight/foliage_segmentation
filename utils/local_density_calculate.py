import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compute_volume_distance(points):
    # TODO: 你之前的 volume distance 公式，这里我先假设返回 (N,)
    vd = np.linalg.norm(points, axis=1)  # 占位
    return vd

def compute_chamfer_distance(points):
    # TODO: 你之前的 chamfer distance 公式，这里我先假设返回 (N,)
    cd = np.linalg.norm(points - points.mean(axis=0), axis=1)  # 占位
    return cd

def analyze_pointcloud(file_path, save_dir):
    points = np.load(file_path)  # shape [N, 3]

    # 计算两个距离
    vd = compute_volume_distance(points)
    cd = compute_chamfer_distance(points)

    stats = {}
    for name, arr in [("volume", vd), ("chamfer", cd)]:
        stats[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

        # 可视化分布
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        sns.histplot(arr, bins=50, kde=True, color="blue")
        plt.title(f"{name} histogram")

        plt.subplot(1,2,2)
        sns.boxplot(x=arr, color="orange")
        plt.title(f"{name} boxplot")

        fname = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{fname}_{name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    return stats

def analyze_directory(dir_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    all_stats = {}

    volume_all, chamfer_all = [], []

    for file in tqdm(os.listdir(dir_path)):
        if not file.endswith(".npy"):
            continue
        fpath = os.path.join(dir_path, file)
        stats = analyze_pointcloud(fpath, save_dir)
        all_stats[file] = stats

        volume_all.extend(np.load(fpath).shape[0] * [stats["volume"]["mean"]])
        chamfer_all.extend(np.load(fpath).shape[0] * [stats["chamfer"]["mean"]])

    # 全局统计
    global_stats = {
        "volume": {
            "mean": float(np.mean(volume_all)),
            "median": float(np.median(volume_all)),
            "std": float(np.std(volume_all)),
            "min": float(np.min(volume_all)),
            "max": float(np.max(volume_all))
        },
        "chamfer": {
            "mean": float(np.mean(chamfer_all)),
            "median": float(np.median(chamfer_all)),
            "std": float(np.std(chamfer_all)),
            "min": float(np.min(chamfer_all)),
            "max": float(np.max(chamfer_all))
        }
    }

    return all_stats, global_stats


if __name__ == "__main__":
    stats, global_stats = analyze_directory(
        dir_path="./data/ForestSemantic_Difficult/tiles_filtered_fps",
        save_dir="./local_density_results_vis"
        )

    print("全局统计：", global_stats)
