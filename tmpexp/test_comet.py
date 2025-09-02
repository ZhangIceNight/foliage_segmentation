# pip install comet_ml
from comet_ml import Experiment

# ------------------------
# 创建 Experiment
# ------------------------
experiment = Experiment(
    project_name="foliage-segmentation",
    workspace="zwjnefu",
    auto_output_logging="simple",   # 关闭过多自动日志
)

# 设置 experiment 名称
experiment.set_name("ForestSemantic_Difficult_Test")

# ------------------------
# 上传一些参数、指标和简单数据
# ------------------------
experiment.log_parameters({
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 1
})

# 模拟训练过程
for step in range(5):
    experiment.log_metric("train_loss", 1.0/(step+1), step=step)
    experiment.log_metric("accuracy", step*0.1, step=step)

# ------------------------
# 上传文件示例
# ------------------------
with open("test_file.txt", "w") as f:
    f.write("Hello Comet!")

experiment.log_asset("test_file.txt")

# ------------------------
# 打印 experiment URL
# ------------------------
print("Experiment URL:", experiment.get_url())
