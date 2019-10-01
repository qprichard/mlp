
def get_dataset():

    f = open("./iris_data/iris.data", 'r')

    lines = f.readlines()
    f.close()

    dataset = {
        'input': [],
        'output': [],
        }
    for line in lines:
        if len(line) > 1:
            x_0, x_1, x_2, x_3, y = line.replace('\n','').split(',')
            dataset['input'].append([float(x_0), float(x_1), float(x_2), float(x_3)])
            
            index= {
                'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2,
            }
            dataset['output'].append(index[y])
    return dataset

get_dataset()
