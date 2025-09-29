# Dummy cell to test CuPy GPU usage
import cupy as cp

# Create a random array on GPU
x = cp.random.rand(1000, 1000)
y = cp.random.rand(1000, 1000)

# Perform a matrix multiplication
z = cp.dot(x, y)

# Print device info and result summary
print("CuPy is using device:", cp.cuda.runtime.getDeviceCount(), "GPU(s)")
print("Array 'z' is on device:", z.device)
print("Sum of z:", cp.sum(z).get())