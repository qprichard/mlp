from mlp import MLP
from iris_data.convert import get_dataset

my_mlp = MLP([4, 2, 3, 3])
input, output = get_dataset()

my_mlp.main(1000, 0.5, np.array(input), np.array(output))
