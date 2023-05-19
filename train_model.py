import torch
import torch.nn as nn
import torch.optim as optim
from model import NeuralNet
from utils import tokenize, bag_of_words

# Define your training data, vocabulary, and tags
# ...

# Define hyperparameters and other training configurations
input_size = len(vocabulary)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (sentence, label) in training_data:
        # Tokenize and preprocess the input sentence
        tokens = tokenize(sentence)
        X = bag_of_words(tokens, vocabulary)
        y = torch.tensor(label, dtype=torch.long)

        # Forward pass
        output = model(X)
        loss = criterion(output, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss at the end of each epoch
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model
data = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": vocabulary,
    "tags": tags,
    "model_state": model.state_dict()
}
torch.save(data, "train.pth")
