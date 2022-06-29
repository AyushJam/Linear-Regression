# Linear-Regression
A project I completed during the course on Fundamentals of Machine Learning on CCBP NxtWave https://10xiitian.ccbp.in/

**Dataset features** :You have the customer's data which includes the time spent on website, duration of membership, time spent on app and session duration along with the yearly amount spent. The dataset **X** has 374 observations with four features and the target variable **Y** is real-valued as expected for linear regression. 
The repo has many files and each of them has a specific function. 
* `predict.py` Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lr.csv". It should be created in the same directory where this code file is present.
* The model is trained in `train.py`.
The final code runs with a Mean Squared Error of 292.2
