# First time setup, run code to install required packages:
# import subprocess
# subprocess.check_call(["pip", "install", "openai"])
# subprocess.check_call(["pip", "install", "pandas"])
# subprocess.check_call(["pip", "install", "tiktoken"])
# subprocess.check_call(["pip", "install", "scipy"])
# subprocess.check_call(["pip", "install", "transformers"])
# subprocess.check_call(["pip", "install", "spacy"])


# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from transformers import GPT2Tokenizer # for counting tokens
import os
import spacy


class ShopifyAssistant:
    def __init__(self):
        self.EMBEDDING_MODEL = "text-embedding-ada-002"
        self.GPT_MODEL = "gpt-3.5-turbo"
        
        filename = "N:\CAREER\SEPEHR\EDUCATION\Brainstation\Data Science\GPT API Keys\OpenAI API.txt"
        with open(filename, 'r') as file:
            openai.api_key = str(file.read().strip())

        embeddings_path = "N:\CAREER\SEPEHR\EDUCATION\Brainstation\Data Science\Deliverables\Hackathon\OpenAI API\Embedding Search-Ask\df_GPT_embedded_final_no_reviews.csv"
        self.df = pd.read_csv(embeddings_path)

        # Initialize spacy and tokenizer
        self.nlp = spacy.blank("en")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Initialize the conversation history
        self.history = [
            {"role": "system", "content": "You are a helpful assistant that aids in Shopify Webstore Development by suggesting apps and providing captions for products."},
        ]

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def query_message(self, query: str, token_budget: int) -> str:
        introduction = "- Assist the merchant with building their Shopify website. \
            - If you are asked a question, IF NECESSARY, ask a question back to make sure you are replying accurately. \
            - If the user asks for recommendations, use the appstore data provided to give 2 recommendations, and write a short summary of each recommendation. \
            - Do not forget to give a recommendation, as you don't want to waste the user's time. \
            - Prioritize free apps.\
            - Put the FULL app name into quotes!\
            - Keep it between 300 characters"
        question = f"\n\nQuestion: {query}"
        message = introduction
        if self.df is not None:
            for _, row in self.df.iterrows():
                text = row['text']
                next_article = f'\n\nText:\n"""\n{text}\n"""'
                if self.count_tokens(message + next_article + question) > token_budget:
                    break
                else:
                    message += next_article
        return message + question

    def ask(self, query: str, model_selected=None, token_budget: int = 4096-100, print_message: bool = False) -> str:
        if model_selected is None:
            model_selected = self.GPT_MODEL

        message = self.query_message(query, token_budget=token_budget)

        if print_message:
            print(message)

        messages = self.history.copy()
        messages.append({"role": "user", "content": message})

        while self.count_tokens(''.join([m['content'] for m in messages])) > 1024:  # Ensure total tokens stay below 1024
            messages.pop(1)  # Remove the oldest messages first (pop(0) is the system message)

        max_tokens = min(1024, token_budget - self.count_tokens(''.join([m['content'] for m in messages])))  # Ensure max_tokens stay below 1024

        response = openai.ChatCompletion.create(
            model=model_selected,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        response_message = response["choices"][0]["message"]["content"]

        # Store the new system message to history
        self.history.append({"role": "assistant", "content": response_message})

        return response_message