import os
import laspy
import open3d as o3d
import numpy as np
from tqdm import tqdm
def read_single_las(file_path):
    """读取单个 LAS 文件为 Open3D 点云对象"""
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到：{file_path}")
        return None
    try:
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}：{e}")
        return None


def split_and_save_tiles(file_path, output_dir, tile_size=1.0, min_points=100):
    """将点云按 tile_size 网格切割并保存为 .npy"""
    os.makedirs(output_dir, exist_ok=True)
    pcd = read_single_las(file_path)
    if pcd is None:
        return
    xyz = np.asarray(pcd.points)
    min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
    tiles = {}
    for point in xyz:
        i = int((point[0] - min_x) // tile_size)
        j = int((point[1] - min_y) // tile_size)
        key = (i, j)
        tiles.setdefault(key, []).append(point)
    print(f"✅ 切割完成，共生成 {len(tiles)} 个格子")
    count = 0
    for key, points in tiles.items():
        if len(points) < min_points:
            continue
        points = np.array(points)
        out_path = os.path.join(output_dir, f"tile_{key[0]}_{key[1]}.npy")
        np.save(out_path, points)
        count += 1
    print(f"✅ 有效 tile 保存完成，共保存 {count} 个")


def split_and_save_tiles_with_labels(file_path, output_dir, tile_size=1.0, min_points=100):
    """
    按 tile_size 对大场景点云切割，并同步切割标签，保存为 .npz 文件。
    
    :param file_path: str, .las 文件路径
    :param output_dir: str, 输出目录
    :param tile_size: float, 网格大小（单位：米）
    :param min_points: int, 小于该点数的 tile 不保存
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 读取坐标和标签
    las = laspy.read(file_path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    labels = las.classification  # 👈 标签

    assert xyz.shape[0] == labels.shape[0], "标签数与点数不一致"

    # Step 2: 初始化切块
    min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
    tiles = {}

    for i, _ in enumerate(range(xyz.shape[0])):
        point = xyz[i]
        label = labels[i]

        ix = int((point[0] - min_x) // tile_size)
        iy = int((point[1] - min_y) // tile_size)
        key = (ix, iy)

        if key not in tiles:
            tiles[key] = {"xyz": [], "label": []}
        tiles[key]["xyz"].append(point)
        tiles[key]["label"].append(label)

    # Step 3: 保存每个 tile（包含标签）
    print(f"✅ 切割完成，共生成 {len(tiles)} 个格子")
    print("正在保存每个 tile...")
    count = 0
    for key, tile_data in tiles.items():
        if len(tile_data["xyz"]) < min_points:
            continue
        xyz_arr = np.array(tile_data["xyz"])
        label_arr = np.array(tile_data["label"], dtype=np.uint8)
        out_path = os.path.join(output_dir, f"tile_{key[0]}_{key[1]}.npz")
        np.savez(out_path, xyz=xyz_arr, label=label_arr)
        count += 1

    print(f"✅ 切割完成，共保存 {count} 个带标签的 tile 到 {output_dir}")


def filter_and_relabel_tiles(input_dir, output_dir, min_points=4096):
    """
    遍历 npz tile 文件，重标 label，并过滤点数不足的文件
    :param input_dir: 输入的 tiles 文件夹路径
    :param output_dir: 输出保存路径
    :param min_points: 最小点数，低于则丢弃该 tile
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .npz 文件
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    total_files = len(npz_files)
    saved_files = 0

    print(f"开始处理 {total_files} 个 tile 文件...")

    for filename in tqdm(npz_files, desc="Processing Tiles"):
        file_path = os.path.join(input_dir, filename)

        try:
            data = np.load(file_path)
            xyz = data['xyz']
            label = data['label'].astype(np.uint8)

            # 选择有效标签的点
            valid_mask = np.isin(label, [2, 3, 4, 5])
            if not np.any(valid_mask):
                # 没有有效点，跳过
                continue

            xyz_valid = xyz[valid_mask]
            label_valid = label[valid_mask]

            # 标签重新映射
            label_valid = np.where(np.isin(label_valid, [2, 3, 4]), 1, 0)

            # 判断点数是否满足
            if len(xyz_valid) < min_points:
                continue

            # 保存文件
            out_path = os.path.join(output_dir, filename)
            np.savez(out_path, xyz=xyz_valid, label=label_valid)
            saved_files += 1

        except Exception as e:
            tqdm.write(f"处理失败: {filename}, 错误: {e}")

    print(f"处理完成，过滤后的 tile 保存在: {output_dir}")
    print(f"共处理 {total_files} 个 tile，保留 {saved_files} 个。")



try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

def _load_points(file_path):
    """内部函数：读取单个点云文件"""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".las", ".laz"]:
        if laspy is None:
            raise ImportError("请先安装 laspy: pip install laspy")
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T
        scale, offset = las.header.scales, las.header.offsets
    elif ext == ".txt":
        with open(file_path, 'r') as f:
            first_line = f.readline()
        
        # 判断首行是否包含非数字字符（简单判断）
        has_header = any(c.isalpha() for c in first_line)
        
        # 根据是否有标题行决定是否跳过首行
        skiprows = 1 if has_header else 0
        try:
            # 先尝试空格/Tab
            points = np.loadtxt(file_path, delimiter=None, usecols=(0, 1, 2), skiprows=skiprows)
        except ValueError:
            # 如果失败，尝试逗号分隔
            points = np.loadtxt(file_path, delimiter=",", usecols=(0, 1, 2), skiprows=skiprows)
        scale, offset = None, None
    elif ext == ".ply":
        if PlyData is None:
            raise ImportError("请先安装 plyfile: pip install plyfile")
        ply = PlyData.read(file_path)
        vertex = ply["vertex"]
        points = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T
        scale, offset = None, None
    elif ext == ".npy":
        points = np.load(file_path)
        if points.ndim > 2:
            points = points.reshape(-1, points.shape[-1])
        if points.shape[1] > 3:
            points = points[:, :3]
        scale, offset = None, None
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    return points, scale, offset


def check_coordinate_unit(path, verbose=True):
    """
    检查点云文件或目录的坐标单位和点密度
    - 单个文件: 返回单位推测
    - 目录: 汇总所有文件，返回平均密度和总点数
    """
    # 如果是目录，收集文件列表
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.splitext(f)[-1].lower() in [".las", ".laz", ".txt", ".ply", ".npy"]]
        if not files:
            raise ValueError("目录中没有可识别的点云文件")

        total_points = 0
        density_list = []
        dx_list, dy_list = [], []

        for f in files:
            points, scale, offset = _load_points(f)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            dx, dy = x.max() - x.min(), y.max() - y.min()
            area = dx * dy if dx > 0 and dy > 0 else 1.0

            density = points.shape[0] / area
            density_list.append(density)
            total_points += points.shape[0]
            dx_list.append(dx)
            dy_list.append(dy)

            if verbose:
                print(f"📂 {os.path.basename(f)}: {points.shape[0]} points, "
                      f"density={density:.2f} pts/m², dx={dx:.2f}, dy={dy:.2f}")

        avg_density = np.mean(density_list)
        max_dx, max_dy = max(dx_list), max(dy_list)

        unit_guess = "unknown"
        if max(max_dx, max_dy) > 10:
            unit_guess = "meter"
        elif max(max_dx, max_dy) > 100:
            unit_guess = "centimeter"
        elif max(max_dx, max_dy) > 1000:
            unit_guess = "millimeter"

        if verbose:
            print("====== 目录统计结果 ======")
            print(f"📊 总点数: {total_points}")
            print(f"📏 平均密度: {avg_density:.2f} pts/m²")
            print(f"🧠 推测单位: {unit_guess}")

        return unit_guess

    else:
        # 单个文件情况
        points, scale, offset = _load_points(path)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        dx, dy = x.max() - x.min(), y.max() - y.min()
        area = dx * dy if dx > 0 and dy > 0 else 1.0
        density = points.shape[0] / area

        if verbose:
            print("📦 File Info:")
            if scale is not None:
                print(f"  Scale:  {scale}")
                print(f"  Offset: {offset}")
            print("📏 坐标范围差值:")
            print(f"  x: {dx:.2f}, y: {dy:.2f}, z: {z.max()-z.min():.2f}")
            print(f"  点数: {points.shape[0]}")
            print(f"  density: {density:.2f} pts/m²")

        unit_guess = "unknown"
        if (scale is not None and scale[0] >= 1.0) or max(dx, dy) > 10:
            unit_guess = "meter"
        elif (scale is not None and scale[0] >= 0.01) or max(dx, dy) > 100:
            unit_guess = "centimeter"
        elif (scale is not None and scale[0] >= 0.001) or max(dx, dy) > 1000:
            unit_guess = "millimeter"

        if verbose:
            print(f"🧠 推测单位: {unit_guess}")

        return unit_guess



def generate_kfold_splits(tile_dir, k=5, output_dir="splits_kfold", seed=42, suffix=".npz"):
    """
    将 tile_dir 中的 tile 文件按 k 折交叉验证划分，输出 train/test 文件列表。
    
    :param tile_dir: str，包含所有 tile_*.npz 的目录
    :param k: int，折数
    :param output_dir: str，保存划分文件的目录
    :param seed: int，随机种子
    :param suffix: str，tile 文件后缀（默认为 .npz）
    """
    os.makedirs(output_dir, exist_ok=True)
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith(suffix)]
    tile_files.sort()  # 保证顺序稳定
    random.seed(seed)
    random.shuffle(tile_files)

    fold_size = len(tile_files) // k
    folds = [tile_files[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    folds.append(tile_files[(k - 1) * fold_size:])  # 最后一折可能稍多一点

    print(f"🔧 总 tile 数量：{len(tile_files)}")
    print(f"📦 每折约 {fold_size} 个 tile")

    for i in range(k):
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        test_files = folds[i]
        train_files = [f for j in range(k) if j != i for f in folds[j]]

        with open(os.path.join(fold_dir, "train.txt"), "w") as f:
            for name in train_files:
                f.write(str(Path(tile_dir) / name) + "\n")

        with open(os.path.join(fold_dir, "test.txt"), "w") as f:
            for name in test_files:
                f.write(str(Path(tile_dir) / name) + "\n")

        print(f"✅ fold_{i}: 训练集 {len(train_files)}，测试集 {len(test_files)}")