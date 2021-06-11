import numpy as np

rng=np.random.default_rng()
algo=rng.choice(20,size=10,replace=False)
# algo=random.sample(range(100), 10)
print(algo)
print("hi")