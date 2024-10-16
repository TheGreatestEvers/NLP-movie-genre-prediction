import torch
from torch.utils.data import Dataset
import random
from nltk.corpus import wordnet  # <-- Add this import to fix the error
from itertools import chain

class MovieGenreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synonym_prob = 0.5  # Probability of applying synonym replacement

    def get_synonym(self, word):
        # Get synonyms using WordNet
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Get lemmas (word forms) from synsets and avoid replacing with the original word
            lemmas = set(chain(*[syn.lemma_names() for syn in synonyms]))
            lemmas.discard(word)
            if lemmas:
                return random.choice(list(lemmas))  # Choose a random synonym
        return word  # If no synonym found, return the original word

    def augment_text_with_synonyms(self, text):
        # Split the text into words
        words = text.split()

        # Randomly replace words with their synonyms
        new_words = []
        for word in words:
            if random.random() < self.synonym_prob:  # Apply synonym replacement with given probability
                synonym = self.get_synonym(word)
                new_words.append(synonym)
            else:
                new_words.append(word)

        # Join the words back into a sentence
        return ' '.join(new_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        augmented_text = self.augment_text_with_synonyms(text)

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )

        input_ids = encoding['input_ids'].flatten()  # Shape: [max_length]
        attention_mask = encoding['attention_mask'].flatten()  # Shape: [max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure labels are tensors
        }

class MovieGenreTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )

        input_ids = encoding['input_ids'].flatten()  # Shape: [max_length]
        attention_mask = encoding['attention_mask'].flatten()  # Shape: [max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }