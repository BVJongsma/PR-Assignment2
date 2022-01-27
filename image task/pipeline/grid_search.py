from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # regularization applied by default
from sklearn.model_selection import GridSearchCV


def grid_search(data, labels):
    knn_model = KNeighborsClassifier()
    param_grid = {'n_neighbors': range(1,15),
         'weights': ['uniform', 'distance']}
    # defining parameter range
    grid = GridSearchCV(knn_model, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

    # fitting the model for grid search
    knn_grid_search = grid.fit(data, labels)
    print(knn_grid_search.best_params_)
    knn_acc = knn_grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(knn_acc))

    lr_model = LogisticRegression()
    param_grid = {'penalty': ['l2', 'none'],
         'max_iter': [100, 500, 1000],
         'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
    # defining parameter range
    grid = GridSearchCV(lr_model, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

    # fitting the model for grid search
    lr_grid_search = grid.fit(data, labels)
    print(lr_grid_search.best_params_)
    lr_acc = lr_grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(knn_acc))

    return knn_model, knn_acc, lr_model, lr_acc