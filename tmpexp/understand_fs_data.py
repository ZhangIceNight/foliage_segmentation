from preprocess import read_single_las
import laspy

las_path = 'data/ForestSemantic_Difficult/Plot_5.las'
las = laspy.read(las_path)
# 查看所有可用的点属性字段
print("所有可用字段：")
print(las.point_format.dimension_names)