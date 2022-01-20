from pipeline import image_loader
from pipeline import feature_extraction
#from pipeline import classification
from pipeline import clustering
from pipeline import grid_search
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    image_path = "BigCats"

    images = image_loader.load_images(image_path) #input: image path, output: images
    #preprocessing(images) zodra we bij de Data augmentation stap zijn
    #reduced_data = feature_extraction(images) #extract features from images
    #model1, model2, model 3 = classification(reduced_data, images) #classify features
    #clustering(model1, model2) #analyse outcome results for original and reduced dataset
    #grid_search() #