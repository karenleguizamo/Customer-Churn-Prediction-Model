Machine learning algorithms used:
We will now take a look at the machine learning algorithms that are implemented. Three ML algorithms are implemented to help us build a robust prediction model. They are as follows:
•	Support Vector Machine
•	Artificial Neural Network
•	XgBoost



Programming logic:
a)	Encoding categorical variables as factors (for SVM and ANN models only):
This is the first data pre-processing step where the categorical variables in the form of characters are converted into numeric factors. Hence the variables ‘France’, ‘Spain’ and ‘Germany’ are replaced by numeric values 1,2 and 3 respectively. Similarly the gender variables ‘Female’ and ‘Male’ are replaced by numbers 1 and 2 respectively. This step helps reduce the computational complexity and improves overall execution speed

b)	Feature Scaling(for SVM and ANN models only): This is the second data pre-processing step where the range of independent variables in the training and test sets are standardized. Most classifiers rely on Euclidean distance to calculate the distance between two points. Feature scaling is done to ensure that each independent variable contributes almost proportionately to the distance.

c)	Applying PCA algorithm:
The Principal Component Analysis algorithm is primarily used for data compression and thereby increase the computation speed of the machine learning algorithm. This algorithm performs dimensionality reduction in datasets which consist of many independent variables corelated with each other. 

d)	Fitting the model to the training set and predicting the test set results:
In case of all the three classifiers used, we first train the model on the training set and use the test set to predict the results

e)	Applying k-Fold cross validation (k=10):
This is another technique that is used to obtain more accurate results. k-fold cross validation is a technique that is used to randomly partition the dataset into k equal sized samples. One of these equally partitioned samples is used as a test set and the other k-1 samples are used as training set.
f)	Computing the accuracies.
