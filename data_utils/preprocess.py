import os
import random
import laspy
import open3d as o3d
import numpy as np
from tqdm import tqdm
import torch
import json
from pathlib import Path
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

def read_single_las(file_path):
    """read single las to Open3d point cloud"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd
    except Exception as e:
        print(f"Read failed: {file_path}: {e}")
        return None


def split_and_save_tiles(file_path, output_dir, tile_size=1.0, min_points=100):
    """
    split point cloud into tiles and save as .npy

    Supported input formats: `.las/.laz`, `.npy`, `.ply`, `.txt`
    - For `.npy`, the first three columns are read as coordinates (if more than 3 columns, truncated to 3)
    - For `.ply/.txt/.las`, x,y,z are automatically read
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        xyz, _, _ = _load_points(file_path)
    except Exception as e:
        print(f"Read failed: {file_path}: {e}")
        return
    min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
    tiles = {}
    for point in xyz:
        i = int((point[0] - min_x) // tile_size)
        j = int((point[1] - min_y) // tile_size)
        key = (i, j)
        tiles.setdefault(key, []).append(point)
    print(f"Split completed, generated {len(tiles)} tiles")
    count = 0
    for key, points in tiles.items():
        if len(points) < min_points:
            continue
        points = np.array(points)
        out_path = os.path.join(output_dir, f"tile_{key[0]}_{key[1]}.npy")
        np.save(out_path, points)
        count += 1
    print(f"Valid tiles saved, total {count} tiles")


def split_and_save_tiles_with_labels(file_path, output_dir, tile_size=1.0, min_points=100):
    """
    Split point cloud into tiles and save labels as well, in `.npz` format.

    Supported input formats: `.las/.laz`, `.npy`, `.ply`
    - `.las/.laz`: Use classification as labels
    - `.npy`: If array has >=4 columns, last column is used as label; first three columns are x,y,z
    - `.ply`: Automatically detect label/class/classification fields in vertex attributes as labels

    :param file_path: str, point cloud file path
    :param output_dir: str, output directory
    :param tile_size: float, tile size (in meters)
    :param min_points: int, tiles with fewer points than this will not be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read coordinates and labels (generic)
    xyz, labels = _load_points_and_labels(file_path)
    if labels is None:
        raise ValueError("This file does not contain usable labels and cannot perform labeled tiling. Please provide a .las/.laz with labels, a .npy with label columns, or a .ply with label attributes.")
    assert xyz.shape[0] == labels.shape[0], "Number of labels does not match number of points"

    # Step 2: Initialize tiles
    min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
    tiles = {}
    print(f"Expected number of tiles: {((xyz[:, 0].max() - min_x) // tile_size + 1) * ((xyz[:, 1].max() - min_y) // tile_size + 1)}")
    for i, _ in tqdm(enumerate(range(xyz.shape[0]))):
        point = xyz[i]
        label = labels[i]

        ix = int((point[0] - min_x) // tile_size)
        iy = int((point[1] - min_y) // tile_size)
        key = (ix, iy)

        if key not in tiles:
            tiles[key] = {"xyz": [], "label": []}
        tiles[key]["xyz"].append(point)
        tiles[key]["label"].append(label)

    # Step 3: Save each tile (including labels)
    print(f"Split completed, generated {len(tiles)} tiles")
    print("Saving each tile...")
    count = 0
    all_point_count = xyz.shape[0]
    unadded_point_count = 0
    added_point_count = 0
    for key, tile_data in tiles.items():
        if len(tile_data["xyz"]) < min_points:
            unadded_point_count += len(tile_data["xyz"])
            print(f"❌ Tile {key} has insufficient points, only: {len(tile_data['xyz'])}, skipped")
            continue
        added_point_count += len(tile_data["xyz"])
        xyz_arr = np.array(tile_data["xyz"])
        label_arr = np.array(tile_data["label"], dtype=np.uint8)
        out_path = os.path.join(output_dir, f"tile_{key[0]}_{key[1]}.npz")
        np.savez(out_path, xyz=xyz_arr, label=label_arr)
        count += 1

    print(f"Save completed, total {count} labeled tiles saved to {output_dir}")
    print(f"all points: {all_point_count}, added points: {added_point_count}, unadded points: {unadded_point_count}")


def filter_and_relabel_tiles(input_dir, output_dir, min_points=4096, relabel=False):
    """
    Traverse npz tile files, relabel, and filter out files with insufficient points
    :param input_dir: Input tiles folder path
    :param output_dir: Output save path
    :param min_points: Minimum points, tiles with fewer points than this will be discarded
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all .npz files
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz') or f.endswith('.npy')]
    total_files = len(npz_files)
    saved_files = 0

    print(f"Starting processing {total_files} tile files...")

    for filename in tqdm(npz_files, desc="Processing Tiles"):
        file_path = os.path.join(input_dir, filename)

        try:
            data = np.load(file_path)
            if filename.endswith('.npy'):
                xyz = data[:, :3]
                label = data[:, -1].astype(np.uint8)
            else:
                xyz = data['xyz']
                label = data['label'].astype(np.uint8)

            if relabel:
                # Select valid points
                valid_mask = np.isin(label, [2, 3, 4, 5]) # ForestSemantic_Difficult
                # valid_mask = np.isin(label, [4, 5]) # Evo
                if not np.any(valid_mask):
                    # No valid points, skip
                    continue

                xyz_valid = xyz[valid_mask]
                label_valid = label[valid_mask]

                # Label remapping
                label_valid = np.where(np.isin(label_valid, [2, 3, 4]), 1, 0) # ForestSemantic_Difficult
                # label_valid = np.where(np.isin(label_valid, [4]), 1, 0) # Evo
                

            else:
                xyz_valid = xyz
                label_valid = label
            # Check if point count meets the requirement
            if len(xyz_valid) < min_points:
                continue

            # Save file
            out_path = os.path.join(output_dir, filename)
            np.savez(out_path, xyz=xyz_valid, label=label_valid)
            saved_files += 1

        except Exception as e:
            tqdm.write(f"Processing failed: {filename}, Error: {e}")

    print(f"Processing completed, filtered tiles saved to: {output_dir}")
    print(f"Total processed {total_files} tiles, retained {saved_files}.")

def farthest_point_sampling_torch(xyz, npoint, device="cuda"):
    """
    Use PyTorch implementation of Farthest Point Sampling (FPS), supports GPU acceleration
    xyz: [N, 3] points (numpy array)
    npoint: object points
    return: [npoint] sampled point indices (numpy array)
    """
    xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance)

    return centroids.cpu().numpy()


def fps_downsample_tiles(input_dir, output_dir, target_points=16384, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    total_files = len(npz_files)
    print(f"Starting FPS downsampling for {total_files} tile files...")

    for filename in tqdm(npz_files, desc="FPS Downsample"):
        file_path = os.path.join(input_dir, filename)

        try:
            data = np.load(file_path)
            xyz = data['xyz']
            label = data['label']

            num_points = len(xyz)

            if num_points <= target_points:
                # Not enough points, keep original
                out_path = os.path.join(output_dir, filename)
                np.savez(out_path, xyz=xyz, label=label)
                continue

            # FPS downsampling (GPU preferred)
            idxs = farthest_point_sampling_torch(xyz, target_points, device=device)

            # Sampled points and labels
            xyz_sampled = xyz[idxs]
            label_sampled = label[idxs]

            # Save
            out_path = os.path.join(output_dir, filename)
            np.savez(out_path, xyz=xyz_sampled, label=label_sampled)

        except Exception as e:
            tqdm.write(f"Downsampling failed: {filename}, Error: {e}")

    print(f"FPS downsampling completed, results saved to: {output_dir}")


def _load_points(file_path):
    """Internal function: Read a single point cloud file"""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".las", ".laz"]:
        if laspy is None:
            raise ImportError("Please install laspy first: pip install laspy")
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T
        scale, offset = las.header.scales, las.header.offsets
    elif ext == ".txt":
        with open(file_path, 'r') as f:
            first_line = f.readline()
        
        # Determine whether the first line contains non-numeric characters (simple check)
        has_header = any(c.isalpha() for c in first_line)
        
        # Decide whether to skip the first row based on the presence of a header row.
        skiprows = 1 if has_header else 0
        try:
            # First try space/Tab
            points = np.loadtxt(file_path, delimiter=None, usecols=(0, 1, 2), skiprows=skiprows)
        except ValueError:
            # If it fails, try comma separation
            points = np.loadtxt(file_path, delimiter=",", usecols=(0, 1, 2), skiprows=skiprows)
        scale, offset = None, None
    elif ext == ".ply":
        if PlyData is None:
            raise ImportError("Please install plyfile first: pip install plyfile")
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
        raise ValueError(f"Unsupported file format: {ext}")
    return points, scale, offset


def _load_points_and_labels(file_path):
    """
    generic loading function: returns (xyz, labels)
    Supports:
    - .las/.laz: uses classification as labels
    - .npy: if columns >= 4, the last column is treated as labels; the first three columns are xyz
    - .ply: if vertex attributes contain label/class/classification fields, they are used as labels
    Other formats do not support returning labels.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".las", ".laz"]:
        if laspy is None:
            raise ImportError("Please install laspy first: pip install laspy")
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        labels = np.asarray(las.classification, dtype=np.uint8)
        return xyz, labels
    elif ext == ".npy":
        arr = np.load(file_path)
        if arr.ndim == 1:
            raise ValueError(".npy requires a 2D array [N, C]")
        if arr.shape[1] < 3:
            raise ValueError(".npy requires at least 3 columns to represent xyz")
        xyz = arr[:, :3]
        labels = arr[:, -1].astype(np.int32) if arr.shape[1] >= 4 else None
        return xyz, labels
    elif ext == ".ply":
        if PlyData is None:
            raise ImportError("Please install plyfile first: pip install plyfile")
        ply = PlyData.read(file_path)
        vertex = ply["vertex"]
        xyz = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T

        if 'gt_class' in vertex:
            labels = np.asarray(vertex['gt_class'])
        else:
            raise ValueError("PLY file not found 'gt_class' label field, unable to extract labels.")
        return xyz, labels
    else:
        # Other formats do not currently support labels
        xyz, _, _ = _load_points(file_path)
        return xyz, None

def check_coordinate_unit(path, verbose=True):
    """
    Check the coordinate unit and point density of a point cloud file or directory.
    - Single file: returns unit guess
    - Directory: summarizes all files, returns average density and total points
    """
    # If it's a directory, collect file list
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.splitext(f)[-1].lower() in [".las", ".laz", ".txt", ".ply", ".npy"]]
        if not files:
            raise ValueError("No recognizable point cloud files found in directory")

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
            print("====== Dir Directory Statistics ======")
            print(f"Total Points: {total_points}")
            print(f"Average Density: {avg_density:.2f} pts/m²")
            print(f"Guessed Unit: {unit_guess}")

        return unit_guess

    else:
        # Single file case
        points, scale, offset = _load_points(path)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        dx, dy = x.max() - x.min(), y.max() - y.min()
        area = dx * dy if dx > 0 and dy > 0 else 1.0
        density = points.shape[0] / area

        if verbose:
            print("File Info:")
            if scale is not None:
                print(f"  Scale:  {scale}")
                print(f"  Offset: {offset}")
            print("Coordinate Range:")
            print(f"  x: {dx:.2f}, y: {dy:.2f}, z: {z.max()-z.min():.2f}")
            print(f"  Points: {points.shape[0]}")
            print(f"  Density: {density:.2f} pts/m²")

        unit_guess = "unknown"
        if (scale is not None and scale[0] >= 1.0) or max(dx, dy) > 10:
            unit_guess = "meter"
        elif (scale is not None and scale[0] >= 0.01) or max(dx, dy) > 100:
            unit_guess = "centimeter"
        elif (scale is not None and scale[0] >= 0.001) or max(dx, dy) > 1000:
            unit_guess = "millimeter"

        if verbose:
            print(f"Guessed Unit: {unit_guess}")

        return unit_guess



def generate_kfold_splits(tile_dir, k=5, output_dir="splits_kfold", seed=42, suffix=".npz"):
    """
    Split files in tile_dir into k-fold cross-validation sets, outputting train/test file lists.

    :param tile_dir: str. dir that contains all tile_*.npz 
    :param k: int, number of folds
    :param output_dir: str, dir to save splited files
    :param seed: int, random seed of splitting
    :param suffix: str, tile file suffix (default: .npz)
    """
    os.makedirs(output_dir, exist_ok=True)
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith(suffix)]
    tile_files.sort()  # make sure the order is stable
    random.seed(seed)
    random.shuffle(tile_files)

    fold_size = len(tile_files) // k
    folds = [tile_files[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    folds.append(tile_files[(k - 1) * fold_size:])  # the last fold may be slightly larger

    print(f"Total tile count: {len(tile_files)}")
    print(f"Each fold has about {fold_size} tiles")

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

        print(f"fold_{i}: training set {len(train_files)}, testing set: {len(test_files)}")



def kfold_split_dataset(input_dir, output_json, k=5, seed=42):
    """
    k-fold cross validation split for point cloud data (.npz)
    - only generate splits.json
    - splits.json format: { "fold_0": {"train": [...], "val": [...]}, ... }
    """
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    npz_files.sort()  # make sure the order is stable

    random.seed(seed)
    random.shuffle(npz_files)

    total = len(npz_files)
    fold_size = total // k
    print(f"Total file count: {total}, each fold size: {fold_size}")

    splits = {}

    for i in range(k):
        val_files = npz_files[i * fold_size : (i + 1) * fold_size]
        train_files = [f for f in npz_files if f not in val_files]

        splits[f"fold_{i}"] = {
            "train": train_files,
            "val": val_files
        }

    with open(output_json, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"{k}-fold cross validation split completed, results saved to: {output_json}")
