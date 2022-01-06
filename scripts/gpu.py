import torch


def describe_torch_CUDA_device(index):
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')


print("Torch CUDA available", torch.cuda.is_available())
print("Torch CUDA device count", torch.cuda.device_count())
print("Torch CUDA current device", torch.cuda.current_device())
print("Torch CUDA version", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    describe_torch_CUDA_device(i)
