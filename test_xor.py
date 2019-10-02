from mlp import MLP

my_mlp = MLP([2, 2, 1])
my_mlp.main(8000, 0.5, np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0]))
