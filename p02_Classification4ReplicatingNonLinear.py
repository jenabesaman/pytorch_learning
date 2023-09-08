import matplotlib.pyplot as plt
import torch

A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(A.dtype, A)
plt.plot(A)
plt.show()

plt.plot(torch.relu(A))
plt.show()


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x)


print(relu(A))
plt.plot(relu(A))
plt.show()

def sigmoid(x):
    return 1 / (1+torch.exp(-x))
plt.plot(sigmoid(A))
plt.show()