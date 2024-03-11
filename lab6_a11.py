import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "C:\\Users\\Anu\\Documents\\SEM_04\\ML\\PROJECT\\Legality prediction\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

# Assuming 'X' contains your feature vectors and 'y' contains your class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'label' is the column containing class labels
y = data['Judgement Status']

# Perform the train-test split with 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the MLPClassifier model
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)

# Make predictions on the testing set
predictions = mlp_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of MLPClassifier:", accuracy)
