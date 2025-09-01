from preprocess import read_single_las
import laspy

las_path = 'data/ForestSemantic_Difficult/Plot_5.las'
las = laspy.read(las_path)

print(las.keys())