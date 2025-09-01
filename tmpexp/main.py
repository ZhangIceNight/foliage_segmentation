from preprocess import check_coordinate_unit

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
    detail_dataset_info(ForestSemantic_Difficult_path, name="ForestSemantic_Difficult")

if __name__ == "__main__":
    Tropical_path = '/public/wjzhang/datasets/LabelledPC'  
    Mixed_path = '/public/wjzhang/datasets/wood_seg_samples/wood_seg_samples' 
    ForestSemantic_path = '/public/wjzhang/datasets/DHMamba_project/ForestSemantic/Plot_1.las' 

    Birch_path = '/public/wjzhang/datasets/Chinese_wood/Birch/reference_pc_White_Birch.npy'  
    Larch_path = '/public/wjzhang/datasets/Chinese_wood/Larch/reference_pc_Dahurian_Larch.npy'  
    Chinese_scholar_tree_path = '/public/wjzhang/datasets/Chinese_wood/Chinese_scholar_tree/reference_pc_Chinese_scholar_tree.npy'  
    Evo_mls_path = '/public/wjzhang/datasets/evonpy/train'  
    ForestSemantic_Difficult_path = '/public/wjzhang/datasets/DHMamba_project/ForestSemantic_Difficult/Plot_5.las'

    main()
