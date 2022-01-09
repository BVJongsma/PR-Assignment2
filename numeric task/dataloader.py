import csv
import numpy as np

class Dataloader():

    def __init__(self):
        self.genes_path = "Genes/"
        reader_data = csv.reader(open(self.genes_path + "data.csv"))
        reader_labels = csv.reader(open(self.genes_path + "labels.csv"))
        self.labels_head = next(reader_labels)
        self.data_header = next(reader_data)[1:]
        self.data = []
        self.labels = []
        for row_data,row_labels in zip(reader_data, reader_labels):
            mapped_to_float = map(float, row_data[1:])
            self.data.append(np.array(list(mapped_to_float)))
            self.labels.append(np.array(row_labels[1]))
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
