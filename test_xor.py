"""XOR learning"""

my_mlp = MLP([2, 2, 1])
my_mlp.main(1000, 0.5, np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[1],[1],[0],[0]]))
