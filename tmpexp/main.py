from preprocess import check_las_coordinate_unit

def main():
    path = '/public/wjzhang/datasets/LabelledPC/11_avec_feuilles_Ref.txt'
    check_las_coordinate_unit(path)

if __name__ == "__main__":
    main()