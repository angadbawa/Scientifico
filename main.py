import os
import warnings
import json
import torch
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

# Import the ScienceBot and utils modules
from scienbot import ScienceBot
from utils import tokenize, bag_of_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

train_data = load_dataset('train.json')
valid_data = load_dataset('valid.json')
test_data = load_dataset('test.json')

# Preprocess the training dataset
def preprocess_dataset(data):
    processed_data = []
    for item in data:
        question = item['question']['stem']
        answer_choices = item['question']['choices']
        correct_answer = item['answerKey']

        processed_data.append({
            'question': question,
            'answer_choices': answer_choices,
            'correct_answer': correct_answer
        })

    return processed_data

processed_train_data = preprocess_dataset(train_data)
processed_valid_data = preprocess_dataset(valid_data)
processed_test_data = preprocess_dataset(test_data)

# Train the model
def train_model():
    # Create a ScienceBot instance
    bot = ScienceBot()

    # Prepare the training data
    all_words = []
    tags = []
    xy = []

    for item in processed_train_data:
        question = item['question']
        answer_choices = item['answer_choices']
        correct_answer = item['correct_answer']

        tags.append(correct_answer)

        for choice in answer_choices:
            tokenized_question = tokenize(question)
            all_words.extend(tokenized_question)
            xy.append((tokenized_question, choice))

    # Perform stemming and create the bag of words
    ignore_words = ['?', '!']
    all_words = [stem(word) for word in all_words if word not in ignore_words]
    all_words = sorted(set(all_words))

    X_train = []
    y_train = []

    for (sentence, choice) in xy:
        bag = bag_of_words(sentence, all_words)
        X_train.append(bag)

        label = tags.index(choice)
        y_train.append(label)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Train the model
    bot.train_model(X_train, y_train)

    # Save the trained model
    model_path = "train.pth"
    bot.save_model(model_path)
    print(f"Trained model saved to {model_path}")

# Callback function called on update config
def config(configuration: ConfigClass):
    # Train the model when the config is updated
    train_model()

# Callback function called on each execution pass
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    # Create a ScienceBot instance
    bot = ScienceBot()

    # Load the trained model
    model_path = "train.pth"
    bot.load_model(model_path)

    output = []
    for text in request.text:
        # Generate the chatbot response
        response = bot.get_response(text)
        output.append(response)

    return SimpleText(dict(text=output))


# Configuration and execution
if __name__ == "__main__":
    configuration = ConfigClass()
    config(configuration)

    request = SimpleText(text=["What is the capital of France?"])
    ray = OpenfabricExecutionRay()
    response = execute(request, ray)

    print(response.text)
