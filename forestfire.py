# ForestFiredetection
 # ForestFiredetection
 import numpy as np
 import pandas as pd
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.preprocessing import LabelEncoder
 
 # Load your dataset
 data = pd.read_csv('/content/forestfires.csv')
 
 # Define a threshold for fire detection (e.g., area > 0)
 fire_threshold = 0  # Adjust as needed
 
 # Create a binary target variable for fire detection
 data['fire_detected'] = (data['area'] > fire_threshold).astype(int)
 
 # Define the target variable
 target_variable = 'fire_detected'
 
 # Separate features (X) and target (y)
 X = data.drop(columns=[target_variable, 'area'])
 y = data[target_variable]
 
 # Convert categorical features to numerical using Label Encoding
 categorical_features = ['month', 'day']
 for feature in categorical_features:
     le = LabelEncoder()
     X[feature] = le.fit_transform(X[feature])
 
 # Number of iterations for simulation
 num_iterations = 10  # You can adjust this
 
 # Store accuracy values for each iteration
 accuracy_values = []
 
 # Perform multiple iterations
 for i in range(num_iterations):
     # Split the data into training and testing sets with different random states
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=i * 42  # Vary random state
     )
 
     # Initialize the Random Forest classifier
     clf = RandomForestClassifier(n_estimators=100, random_state=42)
 
     # Train the model
     clf.fit(X_train, y_train)
 
     # Make predictions
     y_pred = clf.predict(X_test)
 
     # Evaluate the model using accuracy
     accuracy = accuracy_score(y_test, y_pred)
     accuracy_values.append(accuracy)
 
 # Create a line graph for accuracy
 plt.figure(figsize=(8, 6))  # Adjust figure size if needed
 plt.plot(range(1, num_iterations + 1), accuracy_values, marker='o', color='skyblue')
 plt.xlabel('Iteration')
 plt.ylabel('Accuracy')
 plt.title('Fire Detection Accuracy Over Multiple Iterations')
 plt.ylim([0, 1])  # Set y-axis limits for accuracy
 plt.grid(True)  # Add grid for better visualization
 plt.show()
 
 # ... (rest of the code for feature importance plotting remains the same)