"""Iris dataset learning"""

#get the iris dataset
def get_dataset():
    f = open("./iris_data/iris.data")

    lines = f.readlines()
    f.close()

    np.random.shuffle(lines)
    input = []
    output = []
    training_set = {}
    testing_set = {}
    for line in lines:
        if(len(line) > 1):
            x_0, x_1, x_2, x_3, y = line.replace('\n','').split(',')
            input.append([float(x_0),float(x_1),float(x_2),float(x_3)])

            if y == 'Iris-setosa':
                output.append([1,0,0])
            if y == 'Iris-versicolor':
                output.append([0,1,0])
            if y == 'Iris-virginica':
                output.append([0,0,1])

            training_set = {
                'input': np.array(input[0:120]),
                'output': np.array(output[0:120])
            }

            testing_set =  {
                'input': np.array(input[121:151]),
                'output': np.array(output[121:151])
            }

    return (training_set, testing_set)


my_mlp = MLP([4, 2, 3, 3])

training_set, testing_set = get_dataset()

training_err, testing_err = my_mlp.train(1000, 0.05, training_set, testing_set, 10)


my_mlp.graph_error(training_err, testing_err)
