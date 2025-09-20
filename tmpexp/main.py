from preprocess import check_coordinate_unit, split_and_save_tiles_with_labels, filter_and_relabel_tiles, \
fps_downsample_tiles, kfold_split_dataset

def detail_dataset_info(path, name="dataset"):
    print(f"\n===== {name} =====")
    check_coordinate_unit(path, verbose=True)


def main():
    path = dataset_paths[DATASET_NAME]
    output_dir = dataset_output_dirs[DATASET_NAME]



    # detail_dataset_info(path, name="Tropical")
    # detail_dataset_info(path, name="Mixed")
    # detail_dataset_info(ForestSemantic_path, name="ForestSemantic")

    # detail_dataset_info(Birch_path, name="Birch")
    # detail_dataset_info(Larch_path, name="Larch")
    # detail_dataset_info(Chinese_scholar_tree_path, name="Chinese_scholar_tree")
    # detail_dataset_info(path, name="Evo")
    # detail_dataset_info(ForestSemantic_Difficult_path, name="ForestSemantic_Difficult")

    # split_and_save_tiles_with_labels(path, output_dir=output_dir, tile_size=1, min_points=4096)
    # filter_and_relabel_tiles(input_dir=f"./data/{DATASET_NAME}/tiles", output_dir=f"./data/{DATASET_NAME}/tiles_filtered", min_points=4096, relabel=True)
    # fps_downsample_tiles(input_dir=f"./data/{DATASET_NAME}/tiles_filtered", output_dir=f"./data/{DATASET_NAME}/tiles_filtered_fps", target_points=16384)
    kfold_split_dataset(input_dir=f"./data/{DATASET_NAME}/tiles_filtered_fps", output_json=f"./data/{DATASET_NAME}/splits.json", k=5)
if __name__ == "__main__":
    DATASET_NAME = "ForestSemantic_Simple"

    dataset_paths = {
        "Tropical": "/home/wjzhang/workspace/datasets/DHMamba_project/Tropical/tiles",
        "MixedForest": "/home/wjzhang/workspace/datasets/DHMamba_project/MixedForest/tiles",
        "ForestSemantic_Simple": "/public/wjzhang/datasets/DHMamba_project/ForestSemantic_Simple/Plot_1.las",
        "Birch": "/home/wjzhang/workspace/datasets/DHMamba_project/Birch/reference_pc_White_Birch.npy",
        "Larch": "/home/wjzhang/workspace/datasets/DHMamba_project/Larch/reference_pc_Dahurian_Larch.npy",
        "Chinese_scholar_tree": "/home/wjzhang/workspace/datasets/DHMamba_project/Chinese_scholar_tree/reference_pc_Chinese_scholar_tree.npy",
        "Evo": "/home/wjzhang/workspace/datasets/DHMamba_project/Evo/plot_a.ply",
        "ForestSemantic_Difficult": "/home/wjzhang/workspace/datasets/DHMamba_project/ForestSemantic_Difficult/Plot_5.las",
    }

    dataset_output_dirs = {
        "Tropical": "/public/wjzhang/datasets/LabelledPC/tiles",
        "MixedForest": "/home/wjzhang/workspace/datasets/DHMamba_project/MixedForest/tiles",
        "ForestSemantic_Simple": "/public/wjzhang/datasets/DHMamba_project/ForestSemantic_Simple/tiles",
        "Birch": "/home/wjzhang/workspace/datasets/DHMamba_project/Birch/tiles",
        "Larch": "/home/wjzhang/workspace/datasets/DHMamba_project/Larch/tiles",
        "Chinese_scholar_tree": "/home/wjzhang/workspace/datasets/DHMamba_project/Chinese_scholar_tree/tiles",
        "Evo": "/home/wjzhang/workspace/datasets/DHMamba_project/Evo/tiles",
        "ForestSemantic_Difficult": "/home/wjzhang/workspace/datasets/DHMamba_project/ForestSemantic_Difficult/tiles",
    }

    main()
