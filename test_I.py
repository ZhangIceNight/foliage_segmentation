import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
from load_point import load_point_cloud
def compute_covariance_matrix(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    sparse_cov_matrix = csr_matrix(cov_matrix)
    return sparse_cov_matrix

def compute_randomized_svd(matrix, n_components=3):
    u, s, vt = randomized_svd(matrix, n_components=n_components)
    return s

def compute_features(points, k=10, w_L=1.0, w_P=1.0, w_S=1.0):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    features = []
    for i in range(len(points)):
        neighborhood = points[indices[i]]
        cov_matrix = compute_covariance_matrix(neighborhood)
        eigenvalues = compute_randomized_svd(cov_matrix)

        eigenvalues = np.sort(eigenvalues)[::-1]
        lambda_0, lambda_1, lambda_2 = eigenvalues

        total_sum = lambda_0 + lambda_1 + lambda_2
        L = (lambda_0 - lambda_1) / total_sum
        P = (lambda_1 - lambda_2) / total_sum
        S = lambda_2 / total_sum

        # 计算指标 I
        I = w_L * L + w_P * P + w_S * S

        curvature, gradient = curvature_gradient(neighborhood)
        features.append([L, P, S, I, curvature, gradient])
    
    return np.array(features)

def curvature_gradient(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues.sort()
    curvature = (eigenvalues[0] - eigenvalues[1]) / np.sum(eigenvalues)
    gradient = np.linalg.norm(np.gradient(np.mean(centered_points, axis=0)))
    return curvature, gradient

def compute_multiscale_features(points, scales=[10, 20, 30], w_L=1.0, w_P=1.0, w_S=1.0):
    all_features = []
    for scale in scales:
        features = compute_features(points, k=scale, w_L=w_L, w_P=w_P, w_S=w_S)
        all_features.append(features)
    
    combined_features = np.mean(np.stack(all_features, axis=0), axis=0)
    return combined_features

# 示例点集
np.random.seed(42)
# points = np.random.rand(100, 3) * 10
file_path = '/home/zwj/data/work/datas/LabelledPC/1_avec_feuilles_Ref.txt'
points = load_point_cloud(file_path)

# 计算特征
features = compute_multiscale_features(points, scales=[10, 20, 30], w_L=1.0, w_P=1.0, w_S=1.0)

print(f"特征形状: {features.shape}")
print(f"示例特征: {features[:5,3]}")