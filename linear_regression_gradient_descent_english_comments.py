# ### Learning Objectives
# - Understand linear regression using score data based on study hours
# - Understand and apply the concept of gradient descent

# Import libraries for creating the sample dataset
import pandas as pd
import matplotlib.pyplot as plt

# Create the score dataset
data = pd.DataFrame(
    [[2, 20], [4, 40], [8, 80], [9, 90]],
    index=["Student_A", "Student_B", "Student_C", "Student_D"],
    columns=["Study_Hours", "Score"]
)
data

# ### How to Find the Optimal w and b That Minimize MSE
# - MSE: Mean Squared Error, used as the cost function
# - 1. Analytical model based on a mathematical formula
# - 2. Model based on gradient descent

# #### 1. Analytical Model Based on a Mathematical Formula
# - `LinearRegression`
# - A method that finds the optimal linear function at once with relatively little computation using the MSE formula

# Import the model library
from sklearn.linear_model import LinearRegression

# 1. Create the model object
linear_model = LinearRegression()

# In this case, LinearRegression does not require manual hyperparameter tuning
# because it computes the solution directly using a mathematical formula.

# 2. Train the model
linear_model.fit(data[["Study_Hours"]], data["Score"])

# #### Check the Weight and Intercept Values
# - `y = wx + b`

# Check the weight (w) and intercept (b) learned by the model
# Expected relationship: approximately y = 10x + 0
print("Slope / Weight:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# 3. Make a prediction
# Predict the score for a student who studied for 7 hours
linear_model.predict(pd.DataFrame({"Study_Hours": [7]}))

# #### 2. Gradient Descent
# - A method for finding the values of `w` (weight) and `b` (bias/intercept) that minimize the MSE of a linear model
# - It gradually moves toward a linear function with lower error
# - We can visualize how the cost changes as the target value (`w`, `b`) changes
#     - Cost function: a function used to measure how far the hypothesis is from the actual values

# Define the hypothesis function (predicted value)
def h(w, x):
    return w * x + 0

# Define the cost function (MSE)
# ((predicted_value - actual_value) ** 2).mean()

# data   : input values
# target : actual values
# weight : coefficient

def cost(data, target, weight):
    y_pre = h(weight, data)  # Predicted values are produced by the hypothesis function

    mse = ((y_pre - target) ** 2).mean()
    return mse

# Check the error value for a given weight
cost(data["Study_Hours"], data["Score"], 10)  # y = 10x + 0

# Case where the predicted weight is 5
cost(data["Study_Hours"], data["Score"], 5)

# Case where the predicted weight is 15
cost(data["Study_Hours"], data["Score"], 15)

# Case where the predicted weight is 8
cost(data["Study_Hours"], data["Score"], 8)

# Visualize how the cost function changes as the weight (w) changes
cost_list = []  # List to store MSE values
for w in range(1, 20):
    mse = cost(data["Study_Hours"], data["Score"], w)
    cost_list.append(mse)
cost_list

# Plot the cost function (MSE)
plt.plot(range(1, 20), cost_list)
plt.xlabel("w")
plt.ylabel("mse")
plt.show()

# Evaluate the model
linear_model.score(data[["Study_Hours"]], data["Score"])

# In regression, model performance is commonly evaluated with the R² score.

# #### `SGDRegressor` (Stochastic Gradient Descent)
# - A linear regression model in scikit-learn that uses gradient descent
# - It updates weights iteratively instead of solving the equation in one step

# Import the model library
from sklearn.linear_model import SGDRegressor

# 1. Create the model and set hyperparameters
# Key hyperparameters:
# - max_iter: number of weight update iterations
# - eta0: learning rate
# - verbose: print the training process

sgd_model = SGDRegressor(
    max_iter=5000,
    eta0=0.001,
    verbose=1,
    random_state=42
)

# 2. Train the model
sgd_model.fit(data[["Study_Hours"]], data["Score"])

# Norm : the current magnitude of the weight vector
# Bias : intercept term
# As training continues, the model gradually moves in the direction
# that reduces the prediction error.

# Unlike LinearRegression, which computes the solution in one step,
# SGDRegressor starts from random initial values for w and b
# and improves them through repeated updates.

# Check the weight and intercept learned by the SGD model
print("w (weight):", sgd_model.coef_)
print("b (intercept):", sgd_model.intercept_)

# The values may not be exactly 10 and 0.
# Example: y = 9.8x + 1.34

# In regression, small differences between predictions and actual values
# are acceptable as long as the model captures the overall relationship well.

# Both LinearRegression and SGDRegressor aim to find the line
# that best represents the data points.

# How do we judge the "best" line?
# -> It should produce the smallest error.
# -> One common error metric is Mean Squared Error (MSE).
# -> Repeatedly reducing that error is the core idea of gradient descent.

# 3. Make a prediction
sgd_model.predict(pd.DataFrame({"Study_Hours": [7]}))

# Check the R² score
sgd_model.score(data[["Study_Hours"]], data["Score"])

# Final check of the learned weight and intercept
