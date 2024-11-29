import torch
import time
from thop import profile

from pt_mamba import get_model as get_model_mamba
from pt_hmamba import get_model as get_model_hmamba
from pct_seg import get_model as get_model_pct
from pointnet2_sem_seg import get_model as get_model_pointnet2
def test_all_models_metrics(*args):
   print(f"\n{'='*80}")
   # 12表示字符串最小宽度为12个字符,>表示右对齐
   # 例如 'Model' 会占用12个字符的宽度,不足12个字符的部分用空格填充
   print(f"{'Model':12} {'Params(M)':>12} {'FLOPs(G)':>12} {'FPS':>8} {'Memory(MB)':>12} {'Time(s)':>10}")
   print(f"{'-'*80}")
   
   for model_name in args:
      metrics = test_model_metrics(model_name)
      print(f"{model_name:12} {metrics['params']/1e6:>12.2f} {metrics['flops']/1e9:>12.2f} "
            f"{metrics['fps']:>8.2f} {metrics['memory']:>12.2f} {metrics['test_time']:>10.2f}")
   
   print(f"{'='*80}\n")





def test_model_metrics(model_name):
    if model_name == "mamba":
        get_model = get_model_mamba
    elif model_name == "hmamba":
        get_model = get_model_hmamba
    elif model_name == "pct":
        get_model = get_model_pct
    elif model_name == "pointnet2":
        get_model = get_model_pointnet2     
    # 初始化模型
    model = get_model(cls_dim=2).cuda()  # 假设分类数为50
    model.eval()
    metrics = {}
    # 生成测试数据
    batch_size = 2   
    n_points = 4096
    x = torch.randn(batch_size, 3, n_points).cuda().contiguous()
    
    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    metrics['total_params'] = total_params
    # print(f"模型总参数量: {total_params/1e6:.2f}M")
    # 2. 计算FLOPs
    flops, params = profile(model, inputs=(x,))
    flops *= 2  # 将MACs转换为FLOPs (1 MAC = 2 FLOPs)
    metrics['flops'] = flops
    metrics['params'] = params
    # 3. 测试推理速度
    warmup = 2
    test_times = 2

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # 计时
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(test_times):
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / test_times
    fps = batch_size / avg_time
    metrics['fps'] = fps
    metrics['test_time'] = avg_time * 1000 


    # 4. 显存占用
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): 
        _ = model(x)
    memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    metrics['memory'] = memory_allocated
    # print(f"峰值显存占用: {memory_allocated:.2f}MB")

    return metrics
if __name__ == "__main__":
   test_model_metrics()