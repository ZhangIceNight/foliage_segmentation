from preprocess import read_single_las

las_path = 'data/ForestSemantic_Difficult/Plot_5.las'
las_data = read_single_las(las_path)

print(las_data.keys())