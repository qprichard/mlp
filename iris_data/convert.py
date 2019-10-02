#get the iris dataset
def get_dataset():
    f = open("./iris_data/iris.data")

    lines = f.readlines()
    f.close()

    input = []
    output = []

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
    return (input, output)
            
