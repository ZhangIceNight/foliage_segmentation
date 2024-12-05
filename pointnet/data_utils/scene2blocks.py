import numpy as np

def scene2blocks(scene_path, block_size_meters=2.0, stride_meters=1.0):
    """
    将点云场景按照实际物理距离分成若干个blocks
    
    参数:
        scene_path: 场景文件路径
        block_size_meters: 每个block的物理尺寸(米)
        stride_meters: 滑动窗口的步长(米)
    
    返回:
        blocks: 包含多个block的列表，每个block是一个点云数组
    """
    # 加载点云场景数据 - 假设格式为 [N, 3] 或 [N, 6] (xyz + 其他特征)
    scene = np.load(scene_path)
    
    # 获取场景的边界框
    min_bound = np.min(scene[:, :3], axis=0)
    max_bound = np.max(scene[:, :3], axis=0)
    
    # 计算在每个维度上的block数量
    scene_size = max_bound - min_bound
    blocks = []
    
    # 在x和y维度上进行滑动窗口
    x_steps = int((scene_size[0] - block_size_meters) / stride_meters) + 1
    y_steps = int((scene_size[1] - block_size_meters) / stride_meters) + 1
    
    least_points = 10000000
    for i in range(x_steps):
        for j in range(y_steps):
            # 计算当前block的边界
            x_min = min_bound[0] + i * stride_meters
            x_max = x_min + block_size_meters
            y_min = min_bound[1] + j * stride_meters
            y_max = y_min + block_size_meters
            
            # 选择在当前block范围内的点
            mask = (scene[:, 0] >= x_min) & (scene[:, 0] < x_max) & \
                   (scene[:, 1] >= y_min) & (scene[:, 1] < y_max)
            block_points = scene[mask]
            
            # 只保存包含足够多点的block（比如至少100个点）
            if len(block_points) >= 4096:
                blocks.append(block_points)
                if least_points > len(block_points):
                    least_points = len(block_points)
    
    return blocks, least_points

if __name__ == '__main__':
    scene_path = './data/reference_pc_Dahurian_Larch.npy'
    # 每个block 2米，步长1米
    blocks, least_points = scene2blocks(scene_path, block_size_meters=2.0, stride_meters=2.0)
    print(f"总共分成了 {len(blocks)} 个blocks")
    if len(blocks) > 0:
        print(f"每个block的形状示例: {blocks[0].shape}")
        print(f"最少点数: {least_points}")
    np.save('./data/blocks_Dahurian_Larch_2m.npy', blocks)


    blocks, least_points = scene2blocks(scene_path, block_size_meters=1.0, stride_meters=1.0)
    print(f"总共分成了 {len(blocks)} 个blocks")
    if len(blocks) > 0:
        print(f"每个block的形状示例: {blocks[0].shape}")
        print(f"最少点数: {least_points}")
    np.save('./data/blocks_Dahurian_Larch_1m.npy', blocks)

