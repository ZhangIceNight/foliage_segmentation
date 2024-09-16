import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

# 生成更多点
np.random.seed(42)  # 为了结果可重复
num_points = 100  # 生成100个点
points = np.random.rand(num_points, 3) * 10  # 随机生成点，范围在0到10之间

def compute_sparse_covariance_matrix(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    sparse_cov_matrix = csr_matrix(cov_matrix)
    return sparse_cov_matrix

def compute_randomized_svd(matrix, n_components=3):
    u, s, vt = randomized_svd(matrix, n_components=n_components)
    return s

def adaptive_scale_selection(points, k=10):
    # 根据点的密度自适应选择尺度
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    average_distance = np.mean(distances[:, -1])
    scale = average_distance * 1.5  # 自定义尺度调整因子
    return scale

def compute_features_at_scale(points, scale):
    sparse_cov_matrix = compute_sparse_covariance_matrix(points)
    eigenvalues = compute_randomized_svd(sparse_cov_matrix)
    
    lambda_0, lambda_1, lambda_2 = eigenvalues
    linear_feature = (lambda_0 - lambda_1) / (lambda_0 + lambda_1 + lambda_2)
    planar_feature = (lambda_1 - lambda_2) / (lambda_0 + lambda_1 + lambda_2)
    scatter_feature = lambda_2 / (lambda_0 + lambda_1 + lambda_2)
    
    return linear_feature, planar_feature, scatter_feature

def feature_fusion(features_list, stability):
    weights = np.array(stability) / np.sum(stability)
    weighted_features = np.average(features_list, axis=0, weights=weights)
    return weighted_features

def curvature_gradient(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues.sort()
    curvature = (eigenvalues[0] - eigenvalues[1]) / np.sum(eigenvalues)
    gradient = np.linalg.norm(np.gradient(np.mean(centered_points, axis=0)))
    return curvature, gradient

# 计算自适应尺度
scale = adaptive_scale_selection(points)

# 多尺度特征提取
features_scale1 = compute_features_at_scale(points, scale)
features_scale2 = compute_features_at_scale(points, scale * 2)
features_scale3 = compute_features_at_scale(points, scale * 4)

# 稳定性评估（简单示例，实际应用中可能需要更复杂的评估）
stability = [1.0, 0.8, 0.6]  # 示例稳定性

# 加权特征融合
final_features = feature_fusion([features_scale1, features_scale2, features_scale3], stability)

# 计算曲率和梯度
curvature, gradient = curvature_gradient(points)

print(f"最终特征: {final_features}")  # shape: (3,)
print(f"曲率: {curvature}")  # shape: ()
print(f"梯度: {gradient}")  # shape: ()