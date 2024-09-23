import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
from load_point import load_point_cloud
from tqdm import tqdm  # 用于显示进度条

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

def process_files(input_folder, output_folder, scales=[10, 20, 30], w_L=1.0, w_P=1.0, w_S=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的文件名
    file_names = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for file_name in tqdm(file_names, desc="Processing files"):
        file_path = os.path.join(input_folder, file_name)
        points = load_point_cloud(file_path)
        
        # 计算特征
        features = compute_multiscale_features(points, scales=scales, w_L=w_L, w_P=w_P, w_S=w_S)
        
        # 保存特征到新文件
        output_file_path = os.path.join(output_folder, file_name.replace('.txt', '_features.txt'))
        np.savetxt(output_file_path, features, delimiter=',')

# 使用示例
input_folder = '/data/work/datas/LabelledPC'
output_folder = '/data/work/datas//Enhanced_LabelledPC'

process_files(input_folder, output_folder, scales=[10, 20, 30], w_L=1.0, w_P=1.0, w_S=1.0)
