import random
from nn_engine import neural_network
from nn_layer import *


p = neural_network(0,0,2,1,0.5,relu,leaky_relu)


for _ in range(200):
    n1 = np.random.randint(0,2)
    n2 = np.random.randint(0,2)
    p.feedback([n1,n2],[n1 and n2])


print(f"[1,1] → {p.feedforward([1,1])}")
print(f"[1,0] → {p.feedforward([1,0])}")
print(f"[0,1] → {p.feedforward([0,1])}")
print(f"[0,0] → {p.feedforward([0,0])}")

