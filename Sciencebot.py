import random
import json
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize

class ScienceBot:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bot_name = "Science Bot"
        self.model = None
        self.all_words = None
        self.tags = None
        self.load_model(model_path)

    def load_model(self, model_path):
        # Load the trained model and other necessary data
        data = torch.load(model_path)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        self.all_words = all_words
        self.tags = tags

    def process_input(self, sentence):
        # Tokenize the input sentence
        sentence = tokenize(sentence)

        # Convert the sentence to a bag of words vector
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        return X

    def get_response(self, sentence):
        # Process the input and generate the bot's response
        X = self.process_input(sentence)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        if tag == "goodbye":
            return random.choice(["Goodbye!", "See you later!", "Take care!"])

        return "I'm sorry, I can't answer that question at the moment."

    def run_chatbot(self):
        print(f"Let's talk about science! Type 'quit' to exit.")
        while True:
            sentence = input('You: ')
            if sentence == "quit":
                break

            response = self.get_response(sentence)
            print(f"{self.bot_name}: {response}")

# Create an instance of the ScienceBot
bot = ScienceBot("train.pth")

# Run the chatbot
bot.run_chatbot()
