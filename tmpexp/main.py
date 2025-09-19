from preprocess import check_coordinate_unit, split_and_save_tiles_with_labels, filter_and_relabel_tiles, \
fps_downsample_tiles, kfold_split_dataset

def detail_dataset_info(path, name="dataset"):
    print(f"\n===== {name} =====")
    check_coordinate_unit(path, verbose=True)


def main():
    # detail_dataset_info(Tropical_path, name="Tropical")
    # detail_dataset_info(Mixed_path, name="Mixed")
    # detail_dataset_info(ForestSemantic_path, name="ForestSemantic")

    # detail_dataset_info(Birch_path, name="Birch")
    # detail_dataset_info(Larch_path, name="Larch")
    # detail_dataset_info(Chinese_scholar_tree_path, name="Chinese_scholar_tree")
    # detail_dataset_info(Evo_mls_path, name="Evo_mls")
    # detail_dataset_info(ForestSemantic_Difficult_path, name="ForestSemantic_Difficult")

    split_and_save_tiles_with_labels(Birch_path, output_dir=Birch_output_dir, tile_size=1, min_points=4096)
    # filter_and_relabel_tiles(input_dir="./data/ForestSemantic_Difficult/tiles", output_dir="./data/ForestSemantic_Difficult/tiles_filtered", min_points=4096)
    # fps_downsample_tiles(input_dir="./data/ForestSemantic_Difficult/tiles_filtered", output_dir="./data/ForestSemantic_Difficult/tiles_filtered_fps", target_points=16384)
    # kfold_split_dataset(input_dir="./data/ForestSemantic_Difficult/tiles_filtered_fps", output_json="./data/ForestSemantic_Difficult/splits.json", k=5)
if __name__ == "__main__":
    Tropical_path = '/public/wjzhang/datasets/LabelledPC'  
    Mixed_path = '/public/wjzhang/datasets/wood_seg_samples/wood_seg_samples' 
    ForestSemantic_path = '/public/wjzhang/datasets/DHMamba_project/ForestSemantic/Plot_1.las' 

    Birch_path = '/home/wjzhang/workspace/datasets/DHMamba_project/Birch/reference_pc_White_Birch.npy'  
    Larch_path = '/home/wjzhang/workspace/datasets/DHMamba_project/Larch/reference_pc_Dahurian_Larch.npy'  
    Chinese_scholar_tree_path = '/home/wjzhang/workspace/datasets/DHMamba_project/Chinese_scholar_tree/reference_pc_Chinese_scholar_tree.npy'  
    Evo_mls_path = '/home/wjzhang/workspace/datasets/DHMamba_project/Evo_mls/train'  
    ForestSemantic_Difficult_path = '/home/wjzhang/workspace/datasets/DHMamba_project/ForestSemantic_Difficult/Plot_5.las'

    Birch_output_dir = '/home/wjzhang/workspace/datasets/DHMamba_project/Birch/tiles'
    Larch_output_dir = '/home/wjzhang/workspace/datasets/DHMamba_project/Larch/tiles'
    Chinese_scholar_tree_output_dir = '/home/wjzhang/workspace/datasets/DHMamba_project/Chinese_scholar_tree/tiles'
    Evo_mls_output_dir = '/home/wjzhang/workspace/datasets/DHMamba_project/Evo_mls/tiles'
    main()
