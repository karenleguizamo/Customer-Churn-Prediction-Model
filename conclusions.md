# Conclusion:

a)	SVM: SVM with a gaussian kernel yielded a prediction model with a decent accuracy rate. However, due to the computational complexity of the dataset, SVM algorithm had the longest execution time when compared to ANN and XGBoost algorithms.

b)	ANN: An ANN with 2 hidden layers and 5 nodes in each layer proved to be extremely efficient both in terms of the accuracy and minimizing execution time, which makes ANN a very efficient ML algorithm especially in cases where large datasets with high degree computational complexity are involved.

c)	XGBoost: Even though no feature scaling and PCA was implemented while executing this gradient boosting algorithm, it still gives the best accuracy amongst all the classifiers used. It is also to be noted that this algorithm had the least execution time.

Hence, We have successfully created a robust churn prediction model which is able to predict the exit status of a customer in a bank using 3 classifiers with a maximum accuracy of 89.075%.

# References:

1.	https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
2.	https://en.wikipedia.org/wiki/Principal_component_analysis
3.	https://www.kaggle.com/filippoo/deep-learning-az-ann/downloads/Churn_Modelling.csv/1
