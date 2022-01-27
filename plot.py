import matplotlib.pyplot as plt




if __name__ == '__main__':
    plt.figure().suptitle('Class Distribution')

    # y_label = ['Cheetah'] * 38
    # y_label = y_label + ['Jaguar'] * 30
    # y_label = y_label + ['Leopard'] * 31
    # y_label = y_label +['Lion'] * 32
    # y_label = y_label + ['Tiger'] * 39

    plt.bar(['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger'], [38,30,31,32,39])

    #plt.hist(y_label)
    plt.ylabel('Frequency')
    plt.xlabel('Different types of cat')
    plt.savefig('cats_distribution.png')
    plt.show()

