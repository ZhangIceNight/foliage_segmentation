import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import os

def visualize_pointcloud_comparison_auto(points, labels_gt, preds_list, method_names,
                                         zoom_radius=1.0, zoom_box_size=2.0,
                                         save_prefix="PointCloudCompare"):
    """
    自动可视化点云比较：
    - 最左侧两列：GT (绿色/棕色) + GT Error (全灰)。
    - 每个方法两列：左 = 预测图(绿色/棕色)，右 = 错误图(灰=正确, 红=错误)。
    """
    N = points.shape[0]
    n_methods = len(preds_list)

    # 计算局部误差差异，用于选择放大区域
    error_ours = (preds_list[-1] != labels_gt).astype(float)
    error_others_mean = np.mean([(pred != labels_gt).astype(float) for pred in preds_list[:-1]], axis=0)
    error_diff = error_others_mean - error_ours

    tree = KDTree(points)
    local_error_diff = np.zeros(N)
    for i in range(N):
        idx = tree.query_radius(points[i:i+1], r=zoom_radius)[0]
        local_error_diff[i] = error_diff[idx].mean()

    # 选取放大区域中心
    idx_max = np.argmax(local_error_diff)
    center = points[idx_max]
    xmin, ymin, zmin = center - zoom_box_size/2
    xmax, ymax, zmax = center + zoom_box_size/2

    def crop_points(points, labels):
        mask = (points[:,0]>=xmin) & (points[:,0]<=xmax) & \
               (points[:,1]>=ymin) & (points[:,1]<=ymax) & \
               (points[:,2]>=zmin) & (points[:,2]<=zmax)
        return points[mask], labels[mask]

    # 绘制 GT
    def plot_gt(ax, points, labels):
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 0.6, 0.0]  # 绿色
        colors[labels == 1] = [0.6, 0.3, 0.0]  # 棕色
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=10, depthshade=False)
        ax.set_title("GT (Label)", fontsize=12)
        ax.set_axis_off()

    # 绘制 GT Error (全灰)
    def plot_gt_error(ax, points, labels):
        colors = np.ones((len(labels), 3)) * 0.6  # 全灰
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=10, depthshade=False)
        ax.set_title("GT (Error)", fontsize=12)
        ax.set_axis_off()

    # 绘制预测
    def plot_prediction(ax, points, pred_labels, title=""):
        colors = np.zeros((len(pred_labels), 3))
        colors[pred_labels == 0] = [0.0, 0.6, 0.0]  # 绿色
        colors[pred_labels == 1] = [0.6, 0.3, 0.0]  # 棕色
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=10, depthshade=False)
        ax.set_title(title + " (Pred)", fontsize=12)
        ax.set_axis_off()

    # 绘制错误
    def plot_error(ax, points, pred_labels, title=""):
        correct_mask = (pred_labels == labels_gt[:len(pred_labels)])
        colors = np.zeros((len(pred_labels), 3))
        colors[correct_mask] = [0.6, 0.6, 0.6]   # 灰色
        colors[~correct_mask] = [1.0, 0.0, 0.0]  # 红色
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=10, depthshade=False)
        ax.set_title(title + " (Error)", fontsize=12)
        ax.set_axis_off()

    # 总图：GT两列 + 每个方法两列
    fig = plt.figure(figsize=(4*(2 + n_methods*2), 6))

    # GT 部分
    points_crop, labels_crop = crop_points(points, labels_gt)
    ax_gt = fig.add_subplot(1, 2 + n_methods*2, 1, projection='3d')
    plot_gt(ax_gt, points_crop, labels_crop)

    ax_gt_err = fig.add_subplot(1, 2 + n_methods*2, 2, projection='3d')
    plot_gt_error(ax_gt_err, points_crop, labels_crop)

    # 各方法
    for i, (pred, name) in enumerate(zip(preds_list, method_names)):
        points_crop, pred_crop = crop_points(points, pred)

        ax_pred = fig.add_subplot(1, 2 + n_methods*2, i*2+3, projection='3d')
        plot_prediction(ax_pred, points_crop, pred_crop, title=name)

        ax_err = fig.add_subplot(1, 2 + n_methods*2, i*2+4, projection='3d')
        plot_error(ax_err, points_crop, pred_crop, title=name)

    plt.tight_layout()
    save_path = os.path.join("F:/workspace/dhmamba_vis/", f"{save_prefix}_gt_pred_error_compare.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"自动局部放大区域中心：{center}, bounding box: {[xmin,xmax,ymin,ymax,zmin,zmax]}")
    print(f"已保存对比图到 {save_path}")


def batch_visualize(base_dir, pred_dirs, method_names, save_dir):
    """
    遍历 base_dir 下所有 *_point.txt，生成对应的 pdf
    - base_dir: GT 的路径（含 point/label 文件）
    - pred_dirs: 各预测结果的文件夹列表，顺序与 method_names 对应
    - method_names: 方法名称列表
    - save_dir: 保存 pdf 的目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 遍历所有点云文件
    for fname in os.listdir(base_dir):
        if not fname.endswith("_point.txt"):
            continue

        tile_id = fname.replace("_point.txt", "")  # tile_10_15
        print(f"处理 {tile_id} ...")

        # GT 文件
        points_path = os.path.join(base_dir, f"{tile_id}_point.txt")
        labels_gt_path = os.path.join(base_dir, f"{tile_id}_label.txt")

        # 预测文件
        pred_paths = [os.path.join(pred_dir, f"{tile_id}_pred.txt") for pred_dir in pred_dirs]

        # 检查文件是否存在
        if not os.path.exists(points_path) or not os.path.exists(labels_gt_path):
            print(f"⚠️ 跳过 {tile_id}：GT 文件缺失")
            continue
        if not all(os.path.exists(p) for p in pred_paths):
            print(f"⚠️ 跳过 {tile_id}：预测文件缺失")
            continue

        # 加载数据
        points = np.loadtxt(points_path)  # (N, 3)
        labels_gt = np.loadtxt(labels_gt_path).astype(int)
        preds_list = [np.loadtxt(p).astype(int) for p in pred_paths]

        # 保存文件名前缀
        save_str = tile_id

        # 调用可视化函数
        visualize_pointcloud_comparison_auto(
            points=points,
            labels_gt=labels_gt,
            preds_list=preds_list,
            method_names=method_names,
            zoom_radius=1.0,
            zoom_box_size=2.0,
            save_prefix=os.path.join(save_dir, save_str)
        )


if __name__ == "__main__":
    # GT 文件夹
    base_dir = "F:/workspace/dhmamba_vis/ForestSemantic_Difficult_65"

    # 各预测文件夹
    pred_dirs = [
        "F:/workspace/dhmamba_vis/ForestSemantic_Difficult_57",
        "F:/workspace/dhmamba_vis/ForestSemantic_Difficult_60",
        "F:/workspace/dhmamba_vis/ForestSemantic_Difficult_65",
    ]

    method_names = ['Sen-Net', 'PointCloudMamba', 'DHMamba (Ours)']

    save_dir = "F:/workspace/dhmamba_vis/pdf_results"

    batch_visualize(base_dir, pred_dirs, method_names, save_dir)