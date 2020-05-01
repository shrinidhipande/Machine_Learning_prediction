
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt 

# Loading data from generailsed data source.

pfile = open('train_test_data', 'rb')
X_train = pickle.load(pfile)
X_test = pickle.load(pfile)
y_train = pickle.load(pfile)
y_test = pickle.load(pfile)

#train the model & fit the model for prediction
regressor = RandomForestRegressor(n_estimators=90)
regressor.fit(X_train, y_train.ravel())
y_pred = regressor.predict(X_test)

#Calculate the accuracy of the prediction and print it.
#r2 used to get accuracy in regression

accuracy = r2_score(y_test, y_pred)   

print('\naccuracy of random forest regression : {0:.3f}'.format(accuracy*100))

comments = X_test[:, 1]

#ploting graph

labels = ['train', 'test']
plt.scatter(X_train[:, 1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.title('comments vs likes')
plt.xlabel('comments')
plt.ylabel('likes')
plt.legend(labels)
plt.plot(X_test[:,1], y_pred, color='red')
plt.show()
