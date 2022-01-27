""" Code in this file is used to create plots/visuals for the report """
import matplotlib.pyplot as plt

# plot the distribution of class labels for the imagetask.
if __name__ == '__main__':
    plt.figure().suptitle('Class Distribution')

    plt.bar(['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger'], [38, 30, 31, 32, 39])
    plt.ylabel('Frequency')
    plt.xlabel('Different types of cat')
    plt.savefig('cats_distribution.png')
    plt.show()
