import os
import math
import numpy as np
import open3d as o3d
from PIL import Image
import argparse
from tqdm import tqdm


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def cut_img(image):
    """
    Image.crop(left, up, right, below)
    left：Distance of the top left corner from the left boundary
    up：Distance of the top left corner from the upper boundary
    right：Distance of the bottom right corner from the left boundary
    below：Distance of the bottom right corner from the upper boundary
    """
    ImageArray = np.array(image)
    row = ImageArray.shape[0]
    col = ImageArray.shape[1]

    x_left = row
    x_top = col
    x_right = 0
    x_bottom = 0

    for r in range(row):
        for c in range(col):
            if ImageArray[r][c][0] < 255:
                if x_top > r:
                    x_top = r
                if x_bottom < r:
                    x_bottom = r
                if x_left > c:
                    x_left = c
                if x_right < c:
                    x_right = c

    if x_left == row and x_top == col and x_right == x_bottom == 0:
        cropped = image
    else:
        cropped = image.crop((x_left - 1, x_top - 1, x_right + 1, x_bottom + 1))  # (left, upper, right, lower)
    return cropped


# Camera Rotation
def camera_rotation(points, labels, out_path, file_name):
    """修改后的camera_rotation函数,处理npy格式数据"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctrl = vis.get_view_control()

    tmp = 0
    interval = 5.82

    use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    while tmp < 60:
        tmp += 1
        if tmp < 30:
            ctrl.rotate(12 * interval, 0)
        elif 30 <= tmp < 60:
            ctrl.rotate(0, 12 * interval)
        elif 60 <= tmp < 90:
            ctrl.rotate(12 * interval / math.sqrt(2), 12 * interval / math.sqrt(2))
        elif 90 <= tmp < 120:
            ctrl.rotate(12 * interval / math.sqrt(2), -12 * interval / math.sqrt(2))

        if tmp in use_number:
            save_path = os.path.join(out_path, f"{file_name}_{tmp}.png")
            if os.path.exists(save_path):
                continue
            vis.poll_events()
            vis.update_renderer()

            img = vis.capture_screen_float_buffer(True)
            img = Image.fromarray((np.asarray(img) * 255).astype(np.uint8))
            img = cut_img(img)
            img.save(save_path)

    vis.destroy_window()
    del ctrl
    del vis


def projection(path, out_path):
    """修改后的projection函数,处理npy文件"""
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    print(f"找到 {len(files)} 个npy文件")
    
    for file in tqdm(files, desc="处理进度"):
        # 读取npy文件
        data = np.load(os.path.join(path, file))
        points = data[:, :3]
        labels = data[:, -1].astype(np.int32)
        
        # 获取文件名(不含扩展名)
        file_name = file.split('.npy')[0]
        
        # 多角度投影
        camera_rotation(points, labels, out_path, file_name)


def main(config):
    path = config.path
    out_path = config.out_path
    generate_dir(path)
    generate_dir(out_path)
    projection(path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='input')
    parser.add_argument('--out_path', type=str, default='output')
    config = parser.parse_args()

    main(config)
