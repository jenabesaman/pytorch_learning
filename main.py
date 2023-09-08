import torch
import numpy
import matplotlib.pyplot as plt

# import torch
#
# print(torch.cuda.is_available())
#
#
# print(torch.cuda.device_count())
#
#
# print(torch.cuda.current_device())
#
#
# print(torch.cuda.device(0))
#
# print(torch.cuda.get_device_name(0))


#
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))


# from hazm import *
# print(Normalizer().normalize("چه گل های زیبایی."))
# # چه گل‌های زیبایی



# print(torch.__version__)
# scaler=torch.tensor(7)
# print(scaler)
# print(scaler.ndim)
# print(scaler.item())
# print(scaler.shape)

# matrix=torch.tensor([[7,8],[9,5]])
# print(matrix.ndim)
# print(matrix.shape)

# tensor=torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
# print(tensor.ndim)
# print(tensor.shape)
# print(torch.rand(size=(3,4)))
# zeros=torch.zeros(size=(3,4))
# ones=torch.ones(size=(3,4))
# print(zeros,ones)
# print(zeros.dtype)

# RANGE=torch.arange(1,23,step=2)
# print(RANGE)

# one_to_ten=torch.arange(start=1,end=11,step=2)
# one_to_ten_like=torch.ones_like(input=one_to_ten)
# print(one_to_ten_like)

# tensor=torch.tensor([2.0,5.0,20.0],dtype=None)
# tensor2=torch.tensor([2.0,5.0,20.0],dtype=torch.float16,device="cuda")
# print(tensor.dtype,tensor2.dtype,tensor2.device)
# # dtype=precision and speed!
# tensor_float32=tensor2.type(torch.float32)
# print(tensor_float32.dtype)
# multiplication_test=tensor2*tensor_float32
# print(multiplication_test.dtype)
# print(f"shape of tensor2 is:{tensor2.shape}")
# print(f"datatype of tensor2 is:{tensor2.dtype}")
# print(f"device tensor2 is on:{tensor2.device}")

# tensor=torch.tensor([1,2,3],dtype=torch.float16,device="cuda")
# tensor_add=tensor+20
# print(tensor_add)
# tensor_multiply=tensor*10
# print(tensor_multiply)
# tensor_subtract=tensor-10
# print(tensor_subtract)
# tensor_multiply2=torch.mul(input=tensor,other=10)
# print(tensor_multiply2)
# tensor_Add2=torch.add(input=tensor,other=10)
# print(tensor_Add2)

# tensor=torch.tensor([1,2,3])
# print(f"{tensor} * {tensor} Equeals {tensor*tensor}")
# print(f"{tensor} * {tensor} matmul= {torch.matmul(input=tensor,other=tensor)}")
#
# value=0
# for i in range(len(tensor)):
#     value+=tensor[i]*tensor[i]
# print(f"matmul={value}")
# print(f"matmul={tensor @ tensor}")
# tensor2=torch.matmul(input=(torch.rand(size=(2,3))),other=(torch.rand(size=(3,2)) ))
# print(tensor2)

# tensor_a=torch.tensor([[7,8],[10,11],[13,14]])
# tensor_b=torch.tensor([[1,2],[4,5],[6,7]])
#
#
# print(f"original shape of Tensor a ={tensor_a.shape} and shape of tensor b is{tensor_b.shape}")
# print("Output:\n")
# print(f"matmul is{torch.mm(input=tensor_a,mat2=tensor_b.T)}")

# x=torch.arange(0,100,10)
# print(f"min{torch.min(x)},max{torch.max(x)},"
#       f"mean{torch.mean(x,dtype=torch.float16)},"
#       f"sum={torch.sum(x)}")
# print(f"minimom value position:{x.argmin()},"
#       f" maximom value position:{x.argmax()}")

# y=torch.arange(0,9)
# print(f"tensor y shape is: {y.shape}")
# z=y.reshape(1,9)
# print(f"tensor z is: {z} & z hsape is {z.shape}")
# x=z.view(1,9)
# print(f"view y is:{x}")
#
# z[:,0]=5
# print(x,y)
#
# squeezed_z=z.squeeze()
# unsqueezed_z=squeezed_z.unsqueeze(dim=1)
# print(torch.stack([x,x,x,x],dim=1))
# print(f"\n z is {z} and squeeze is {z.squeeze()}")
# print(f"\n z shape is: {z.shape} and squeeze shape is: "
#       f"{z.squeeze().shape} & unsqueeze is: "
#       f"{unsqueezed_z}")

# x_original=torch.rand(size=(224,224,3))
# x_permuted=x_original.permute(2,0,1)
# print(f"original tensor shape= {x_original.shape} and "
#       f"premuted tensor shape= {x_permuted.shape}")
#
# x_permuted[0,0,0]=1111
# print(f"x_premuted= {x_permuted[0,0,0]}"
#       f" and x_original= {x_original[0,0,0]}")

# tensor=torch.arange(1,10).reshape(1,3,3)
# print(tensor)
# print(tensor[0,2,2])

# array=numpy.arange(1.0,8.0)
#
# tensor=torch.arange(1,10)
# tensor1=torch.from_numpy(array)
# print(array.dtype,tensor.dtype,tensor1.dtype)

# RANDOM_SEED = 42
# torch.manual_seed(seed=RANDOM_SEED)
# tensor_random_a = torch.rand(size=(3, 4))
# torch.manual_seed(seed=RANDOM_SEED)
# tensor_random_b = torch.rand(size=(3, 4))
# print(tensor_random_a == tensor_random_b)

# print(torch.cuda.is_available())
# device="cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(torch.cuda.device_count())

# device="cuda" if torch.cuda.is_available() else "cpu"
# tensor_a=torch.tensor([1,2,3])
# tensor_b=torch.tensor([4,5,6],device=device)
# tensor_a_GPU=tensor_a.to(device)
# print(f"{tensor_a} {tensor_b} {tensor_a_GPU}")
# tensor_back_on_cpu=tensor_b.cpu().numpy()
# print(tensor_back_on_cpu)
#
# random_seed=1234
# torch.cuda.manual_seed(seed=random_seed)
# random_tensor=torch.rand(size=(7,7),device="cuda")
# print(random_tensor)
# torch.cuda.manual_seed(seed=random_seed)
# tensor_b=torch.rand(size=(1,7),device="cuda")
# tensor_b_transposed=tensor_b.T
# multiplication=torch.matmul(input=random_tensor,other=tensor_b_transposed)
# print(multiplication)

# random_seed=1234
# device="cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(seed=random_seed)
# tensor_a=torch.rand(size=(2,3))
# torch.manual_seed(seed=random_seed)
# tensor_b=torch.rand(size=(2,3))
# tensor_a_gpu=tensor_a.to(device)
# tensor_b_gpu=tensor_b.to(device)
# print(f"tensor a= {tensor_a_gpu} \n"
#       f"tensor b= {tensor_b_gpu}")
# print(f"shape are: tensor a shape: {tensor_a_gpu.shape} \n"
#       f"and tensor b shape: {tensor_b_gpu.shape}")
# multiplication=torch.matmul(input=tensor_a_gpu,other=tensor_b_gpu.T)
# print(f"multiplication= {multiplication}")
# print(f"maximum is: {torch.max(multiplication)} \n "
#       f"and minimum is: {torch.min(multiplication)}")
# print(f"maximum index value is: {torch.argmax(multiplication)} \n "
#       f"minimum index value is: {torch.argmin(multiplication)}")

# torch.manual_seed(7)
# tensor_a=torch.rand(size=(1,1,1,10))
# tensor_a_squeezed=torch.squeeze(tensor_a)
# print(f"tensor a:{tensor_a} and shape:{tensor_a.shape} \n"
#       f"tensor squeezed:{tensor_a_squeezed} and shape:{tensor_a_squeezed.shape}")
