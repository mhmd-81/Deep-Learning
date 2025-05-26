import matplotlib.pyplot as plt
import numpy as np

def plot_model_output(model,X,y):
    if X.shape[1] != 2:
        raise ValueError(f"X must have exactly 2 features for plotting. Got shape: {X.shape}") # check data shape
    X_min, X_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(X_in)

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        print('Doing multiclass classification')
        y_pred = np.argmax(y_pred, axis=1)
    else:
        print('Doing binary classification')
        y_pred = np.round(y_pred).flatten()

    y_pred = y_pred.reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()