import tensorflow as tf

print("GPU Support: ", tf.test.is_built_with_gpu_support())
print("CUDA Support: ", tf.test.is_built_with_cuda())
print("GPUs available:\n", tf.config.list_physical_devices('GPU'))
tf.test.gpu_device_name()
