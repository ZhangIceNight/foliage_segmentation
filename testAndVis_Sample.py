import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from load_point import load_point_cloud

def density_based_sampling(points, k=10, percentage=0.5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    density = np.mean(distances, axis=1)
    num_samples = int(len(points) * percentage)
    selected_indices = np.argsort(density)[-num_samples:]
    return points[selected_indices]

def farthest_point_sampling(points, num_samples):
    N = points.shape[0]
    centroids = np.zeros((num_samples,), dtype=int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        centroids[i] = farthest
        centroid_point = points[farthest, :].reshape(1, -1)
        dist = np.sum((points - centroid_point) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return points[centroids]

def visualize_point_cloud(points, title="Point Cloud", color='b'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1)
    ax.set_title(title)
    plt.show()

# 示例点云数据
N = 10000  # 假设有 10000 个点
points = np.random.rand(N, 3)  # 随机生成点云
file_path = '../../../data/work/datas/LabelledPC/1_avec_feuilles_Ref.txt'
points = load_point_cloud(file_path)
# 可视化原始点云
visualize_point_cloud(points, title="Original Point Cloud", color='b')

# 第一步：基于密度的采样，保留 50% 的点
percentage = 0.5
density_sampled_points = density_based_sampling(points, k=10, percentage=percentage)

# 可视化基于密度采样的点云
visualize_point_cloud(density_sampled_points, title="Density-Based Sampled Point Cloud", color='g')

# 第二步：对密度采样后的点云进行 FPS，进一步采样 2000 个点
final_sampled_points = farthest_point_sampling(density_sampled_points, num_samples=2000)

# 可视化经过 FPS 采样后的点云
visualize_point_cloud(final_sampled_points, title="FPS Sampled Point Cloud", color='r')

# 输出结果
print(f"原始点数: {points.shape[0]}, 基于密度采样后的点数: {density_sampled_points.shape[0]}, FPS 采样后的点数: {final_sampled_points.shape[0]}")
