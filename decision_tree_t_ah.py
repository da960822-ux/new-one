# Prepare input features and target label
x = data.drop(['Comfortable'], axis=1)
y = data['Comfortable']

# Remove the 'RH' column and keep only 'T' and 'AH' as input features
x_mod = x.drop(['RH'], axis=1)

# Display the shape of the processed feature set and target
print(x_mod.shape, y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x_mod,
    y,
    test_size=0.3,
    random_state=2025,
    stratify=y
)

# Initialize the Decision Tree model
x_tree = DecisionTreeClassifier()

# Fit the model to the training data
x_tree.fit(X_train, y_train)

# Measure the model's performance on the test set
test_score = x_tree.score(X_test, y_test)
print("Model Accuracy:", test_score)

# Plot the Decision Tree for interpretation
plt.figure(figsize=(25, 8))
plot_tree(
    x_tree,
    feature_names=X_train.columns,
    class_names=['Uncomfortable', 'Comfortable'],
    filled=True
)
plt.show()
