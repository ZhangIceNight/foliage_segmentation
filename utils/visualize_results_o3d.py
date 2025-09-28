import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

def visualize_pointcloud_comparison_auto(points, labels_gt, preds_list, method_names,
                                         zoom_radius=1.0, zoom_box_size=2.0,
                                         save_prefix="PointCloudCompare", output_combined=True):
    """
    自动可视化点云比较，正确点蓝色，错误点红色，并突出局部优势区域。
    """
    N = points.shape[0]
    n_methods = len(preds_list)
    
    # 计算局部误差差异，用于选择放大区域
    error_ours = (preds_list[-1] != labels_gt).astype(float)  # 假设最后一个是我们的方法
    error_others_mean = np.mean([(pred != labels_gt).astype(float) for pred in preds_list[:-1]], axis=0)
    error_diff = error_others_mean - error_ours

    tree = KDTree(points)
    local_error_diff = np.zeros(N)
    for i in range(N):
        idx = tree.query_radius(points[i:i+1], r=zoom_radius)[0]
        local_error_diff[i] = error_diff[idx].mean()
    
    # 自动选取局部区域
    idx_max = np.argmax(local_error_diff)
    center = points[idx_max]
    xmin, ymin, zmin = center - zoom_box_size/2
    xmax, ymax, zmax = center + zoom_box_size/2

    def crop_points(points, pred_labels):
        mask = (points[:,0]>=xmin) & (points[:,0]<=xmax) & \
               (points[:,1]>=ymin) & (points[:,1]<=ymax) & \
               (points[:,2]>=zmin) & (points[:,2]<=zmax)
        return points[mask], pred_labels[mask]

    # 可视化函数：蓝色=正确，红色=错误
    def plot_points(ax, points, pred_labels, title="", is_gt=False):
        if is_gt:
            colors_plot = np.tile([0,0,1], (len(pred_labels),1))  # 全蓝
        else:
            correct_mask = (pred_labels == labels_gt[:len(pred_labels)])
            colors_plot = np.zeros((len(pred_labels),3))
            colors_plot[correct_mask] = [0,0,1]  # 蓝
            colors_plot[~correct_mask] = [1,0,0] # 红
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors_plot, s=10, depthshade=False)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    # 单图：只显示我们的方法
    points_crop, labels_crop = crop_points(points, preds_list[-1])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    plot_points(ax, points_crop, labels_crop, title=method_names[-1])
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{method_names[-1]}_zoom.pdf", dpi=300)
    plt.close()

    if output_combined:
        # 对比图，第一列 GT
        preds_list_plot = [labels_gt] + preds_list
        method_names_plot = ["GT"] + method_names

        fig = plt.figure(figsize=(4*len(preds_list_plot),6))
        for i, (pred, name) in enumerate(zip(preds_list_plot, method_names_plot)):
            points_crop, labels_crop = crop_points(points, pred)
            ax = fig.add_subplot(1, len(preds_list_plot), i+1, projection='3d')
            plot_points(ax, points_crop, labels_crop, title=name, is_gt=(name=="GT"))
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_all_methods_zoom.pdf", dpi=300)
        plt.close()

    print(f"自动局部放大区域中心：{center}, bounding box: {[xmin,xmax,ymin,ymax,zmin,zmax]}")

if __name__ == "__main__":
    # ----------- 模拟数据 -----------
    N = 4096
    points = np.random.rand(N, 3) * 10
    labels_gt = np.random.randint(0, 2, N)

    n_methods = 10
    preds_list = [np.random.randint(0, 2, N) for _ in range(n_methods)]
    method_names = [
        'LeWoS', 'PointNeXt', 'PointNet++', 'PointTransformer', 
        'PointTransformerV2', 'PointCloudMamba', 'Mamba3D', 
        'Sen-Net', 'DHMamba w/o MSE', 'DHMamba (Ours)'
    ]

    visualize_pointcloud_comparison_auto(
        points=points,
        labels_gt=labels_gt,
        preds_list=preds_list,
        method_names=method_names,
        zoom_radius=1.0,
        zoom_box_size=2.0,
        save_prefix="ExamplePointCloud",
        output_combined=True
    )
