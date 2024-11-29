import torch
import time
from thop import profile
from ptflops import get_model_complexity_info
from models.pt_hmamba import get_model
def test_model_metrics():
   # 初始化模型
   model = get_model(cls_dim=2).cuda()  # 假设分类数为50
   model.eval()
   
   # 生成测试数据
   batch_size = 2
   n_points = 2048
   x = torch.randn(batch_size, 3, n_points).cuda()
   
   # 1. 计算参数量
   total_params = sum(p.numel() for p in model.parameters())
   
   # 2. 计算FLOPs
   flops, params = profile(model, inputs=(x,))
   flops *= 2  # 将MACs转换为FLOPs (1 MAC = 2 FLOPs)
   
   # 3. 测试推理速度
   warmup = 10
   test_times = 100
   
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
   
   # 打印结果
   print(f"模型总参数量: {total_params/1e6:.2f}M")
   print(f"FLOPs: {flops/1e9:.2f}G")
   print(f"参数量 (from thop): {params/1e6:.2f}M")
   print(f"平均推理时间: {avg_time*1000:.2f}ms")
   print(f"FPS: {fps:.2f}")
   
   # 4. 显存占用
   torch.cuda.reset_peak_memory_stats()
   with torch.no_grad():
       _ = model(x)
   memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
   print(f"峰值显存占用: {memory_allocated:.2f}MB")
if __name__ == "__main__":
   test_model_metrics()