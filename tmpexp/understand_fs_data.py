from preprocess import read_single_las
import laspy
import numpy as np

las_path = 'data/ForestSemantic_Difficult/Plot_5.las'
las = laspy.read(las_path)
xyz = las.xyz
point_class = las.classification
point_source_id = las.point_source_id

print(f"点云总点数: {xyz.shape[0]}")
print(np.unique(point_class))
print(np.unique(point_source_id))