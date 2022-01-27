# PR-Assignment2

Assignment as part of the course Pattern-Recognition-AI at the University of Groningen

For the image dataset (task 1) and task 2, our program was coded using python. For the numerical dataset, our program was coded using Jupyter and python.

In order to run the code, some python packages need to be installed first. These can be found in requirements.txt.
The packages can be installed with the command: pip install -r requirements.txt


To run the pipeline of the numerical task,

To run the pipeline of the image task, you can run the file `Task_1_Image\main.py` using python (tested using python 3.7).
The parameters at the beginning of the file (lines 22 until 28) determine what part of the pipeline is being run
image_path describes the path to the images. If this has not been changed in your installation please keep this as is.
augment determines whether or not to use augmented data. Do note that if you do not yet have these that they will be generated first, this will require at least 61 MB of free space.
extraction_method is a string defining which extraction method is used for feature selection. This can be either 'sift' or 'orb'.
If feature_optimization is true then our pipeline runs tests to check the accuracy of our models with varying amounts of features for our extraction methods
If grid_search is True then grid search will be applied to our models to determine which hyper-parameters yield the highest accuracy
If classification_and_clustering is true, classification and clustering steps will be run for the amount of iterations given in range_limit
range_limit determines how many iterations of tests will be applied, csv files containing the accuracy per iteration will be saved in the csv folder.

To run the code of task 2, run the file `Task_2\main.py` using python (tested using python 3.7).
