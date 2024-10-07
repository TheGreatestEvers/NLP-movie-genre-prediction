from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import GenrePredictor
from data import Dataset
from tqdm import tqdm
import torch
import pandas as pd
from data import MovieGenreDataset
from sklearn.model_selection import train_test_split
import wandb

if torch.backends.mps.is_available():
    device = torch.device("mps")  # For Apple Silicon
    print("Selected device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")  # For NVIDIA GPUs
    print("Selected device: cuda")
else:
    device = torch.device("cpu")    # Fallback to CPU
    print("Selected device: cpu")

def train(num_epochs=100, data_path = "./data/train_combined.csv"):
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Read dataframe and map genres to class numbers
    data = pd.read_csv(data_path).head(10)
    genre_mapping = {
        'drama': 0,
        'comedy': 1,
        'horror': 2,
        'action': 3,
        'romance': 4,
        'western': 5,
        'animation': 6,
        'crime': 7,
        'sci-fi': 8
    }
    data['genre_numbers'] = data['genre'].map(genre_mapping)

    # Split the data
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    texts_train = train_data["combined_text"].reset_index(drop=True)
    labels_train = train_data["genre_numbers"].reset_index(drop=True)
    train_dataset = MovieGenreDataset(texts_train, labels_train, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    texts_val = val_data["combined_text"]
    labels_val = val_data["genre_numbers"]
    val_dataset = MovieGenreDataset(texts_val, labels_val, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    texts_test = test_data["combined_text"]
    labels_test = test_data["genre_numbers"]

    # Initialize training related
    model = GenrePredictor(num_genres=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    wandb.init(
        project="nlp_movie_genres",
    )

    # Train loop
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            wandb.log({"train_batch_loss": loss.item()})
        
        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch: {epoch}, training loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()