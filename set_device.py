import torch

def find_free_gpu():
    device = None
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        for i in range(num_devices):
            if i != current_device:
                if torch.cuda.get_device_capability(i)[0] >= 3.5:  # 可根据你的需求选择支持的计算能力
                    torch.cuda.set_device(i)
                    allocated_memory = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    if allocated_memory == 0 and max_memory == 0:
                        device = torch.device(f"cuda:{i}")
                        break
    return device

# 查找空闲的GPU设备
device = find_free_gpu()
if device is not None:
    print("空闲GPU设备已找到，使用设备:", device)
    # 在训练过程中可以将模型移动到该设备
    # model.to(device)
else:
    print("没有找到空闲的GPU设备，将使用CPU进行训练")
    # 在训练过程中可以使用CPU
    # device = torch.device("cpu")
    # model.to(device)