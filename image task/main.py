from pipeline import image_loader
from pipeline import feature_extraction
#from pipeline import classification
from pipeline import clustering
#from pipeline import grid_search
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    image_path = "BigCats"
    augment = True;

    if (augment):
        image_loader.augment_images(image_path)

    #images = image_loader.load_images(image_path) #input: image path, output: images
    #preprocessing(images) zodra we bij de Data augmentation stap zijn
    #reduced_data = feature_extraction(images) #extract features from images
    #model1, model2, model3 = classification(images, images_label) #classify features
    #rmodel1, rmodel2, rmodel3 = classification(reduced, reduced_label) #classify features reduced images
    #clustering(model1, model2) #analyse outcome results for original and reduced dataset
    #grid_search() #