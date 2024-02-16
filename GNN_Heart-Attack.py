import random
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("ga+adam.log"),
                              logging.StreamHandler()])

class ANN(Sequential):
    def __init__(self, child_weights=None):
        super().__init__()

        if child_weights is None:
            layer1 = Dense(5, input_shape=(5,), activation='relu')
            layer2 = Dense(4, activation='sigmoid')
            layer3 = Dense(2, activation='sigmoid')
            layer4 = Dense(1, activation='sigmoid')
            self.add(layer1)
            self.add(layer2)
            self.add(layer3)
            self.add(layer4)
        else:
            self.add(
                Dense(
                    5,
                    input_shape=(5,),
                    activation='relu',
                    weights=[child_weights[0], np.ones(5)])
                )
            self.add(
                Dense(
                    4,
                    activation='sigmoid',
                    weights=[child_weights[1], np.zeros(4)])
            )
            self.add(
                Dense(
                    2,
                    activation='sigmoid',
                    weights=[child_weights[2], np.zeros(2)])
            )
            self.add(
                Dense(
                    1,
                    activation='sigmoid',
                    weights=[child_weights[3], np.zeros(1)])
            )

    def forward_propagation(self, train_feature, train_label):
        predict_label = self.predict(train_feature)
        self.fitness = accuracy_score(train_label, predict_label.round())

    def compile_train(self, epochs, train_feature, train_label):
        # Normalize the data
        scaler = MinMaxScaler()
        train_feature = scaler.fit_transform(train_feature)

        self.compile(
                      optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss='mean_squared_error',
                      metrics=['accuracy']
                      )
        history = self.fit(train_feature, train_label, epochs=epochs, verbose=0)
        self.scaler = scaler
        return history

def crossover(nn1, nn2):
    nn1_weights = []
    nn2_weights = []
    child_weights = []

    for layer in nn1.layers:
        nn1_weights.append(layer.get_weights()[0])

    for layer in nn2.layers:
        nn2_weights.append(layer.get_weights()[0])

    for i in range(len(nn1_weights)):
        split = random.randint(0, nn1_weights[i].shape[1] - 1)
        for j in range(split, nn1_weights[i].shape[1]):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]
        child_weights.append(nn1_weights[i])

    mutation(child_weights)

    child = ANN(child_weights)
    return child

def mutation(child_weights):
    selection = random.randint(0, len(child_weights) - 1)
    mut = random.uniform(0, 1)
    if mut <= .05:
        child_weights[selection] *= random.uniform(2, 5)
    else:
        pass
    
# Preprocess Data
df = pd.read_table('./heart.txt',header=None,encoding='gb2312',sep='\t')

# Separate features and labels
features = df.iloc[:, :5]  # First 5 columns are features
labels = df.iloc[:, 5]     # Last column is the label

# Convert the first two features to integer
features.iloc[:, 0:2] = features.iloc[:, 0:2].astype(int).copy()

# Convert the remaining features to binary (0 or 1)
features.iloc[:, 2:5] = features.iloc[:, 2:5].applymap(lambda x: 1 if x > 0 else 0).copy()

# Convert labels to integers (0 or 1)
labels = labels.apply(lambda x: 1 if x > 0 else 0)

# Fit the scaler on the entire dataset
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
train_feature = features_scaled[:160]
train_label = labels.iloc[:160]

test_feature = features_scaled[160:]
test_label = labels.iloc[160:]

# store all active ANNs
networks = []
pool = []
# Generation counter
generation = 0
# Initial Population
population = 20
for i in range(population):
    networks.append(ANN())
# Track Max Fitness
max_fitness = 0
# Store Max Fitness Weights
optimal_weights = []

epochs = 20  # Set the number of epochs to 10
losses = []
accuracies = []
# Evolution Loop
for i in range(epochs):
    generation += 1
    logging.info("Generation: " + str(generation) + "\r\n")

    for ann in networks:
        # Propagate to calculate fitness score
        ann.forward_propagation(train_feature, labels.iloc[:160])
        # Add to pool after calculating fitness
        pool.append(ann)

    # Clear for propagation of next children
    networks.clear()

    # Sort anns by fitness
    pool = sorted(pool, key=lambda x: x.fitness)
    pool.reverse()

    # Find Max Fitness and Log Associated Weights
    for i in range(len(pool)):
        if pool[i].fitness > max_fitness:
            max_fitness = pool[i].fitness

            logging.info("Max Fitness: " + str(max_fitness) + "\r\n")

            # Iterate through layers, get weights, and append to optimal
            optimal_weights = []
            for layer in pool[i].layers:
                optimal_weights.append(layer.get_weights()[0])
            logging.info('optimal_weights: ' + str(optimal_weights) + "\r\n")

    # Crossover: top 5 randomly select 2 partners
    for i in range(5):
        for j in range(2):
            # Create a child and add to networks
            temp = crossover(pool[i], random.choice(pool))
            # Add to networks to calculate fitness score next iteration
            networks.append(temp)

# Create a Genetic Neural Network with optimal initial weights
ann = ANN(optimal_weights)
history = ann.compile_train(epochs=20, train_feature=train_feature, train_label=labels.iloc[:160])

# Plot Loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Evaluate the model on the test data
predict_label = ann.predict(test_feature)
print('Test Accuracy: %.2f' % accuracy_score(test_label, predict_label.round()))

# Function to process user input and make predictions
def process_input():
    user_input = entry.get().strip()
    
    # Check if user wants to quit
    if user_input.lower() == 'q':
        root.quit()
        return
    
    # Convert user input to array
    try:
        user_input_array = np.array([int(x) if i < 2 else int(x) if int(x) == 0 or int(x) == 1 else float(x) for i, x in enumerate(user_input.split())], dtype=np.float32).reshape(1, -1)
        user_input_array[:, 2:5] = user_input_array[:, 2:5].astype(int)

        # Normalize the user input using the same scaler
        user_input_normalized = scaler.transform(user_input_array)

        # Predict the output for normalized user input
        predict_label = ann.predict(user_input_normalized)[0][0]

        # Display predicted label
        messagebox.showinfo("Prediction", f"Predicted label: {predict_label}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create GUI window
root = tk.Tk()
root.title("Neural Network Prediction")

# Label and entry for user input
tk.Label(root, text="Enter 5 input values separated by space (integer for the first two inputs and binary for the rest):").pack()
entry = tk.Entry(root, width=50)
entry.pack()

# Button to process input
tk.Button(root, text="Predict", command=process_input).pack()

# Run the GUI
root.mainloop()
