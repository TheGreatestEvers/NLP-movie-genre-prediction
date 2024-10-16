from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoTokenizer
from model import GenrePredictor
from data import Dataset
from tqdm import tqdm
import torch
import pandas as pd
from data import MovieGenreDataset, MovieGenreTestDataset
from sklearn.model_selection import train_test_split
import wandb
import torch.optim as optim

if torch.backends.mps.is_available():
    device = torch.device("mps")  # For Apple Silicon
    print("Selected device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")  # For NVIDIA GPUs
    print("Selected device: cuda")
else:
    device = torch.device("cpu")    # Fallback to CPU
    print("Selected device: cpu")

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)  # Get predictions by taking argmax over the logits
    correct = (preds == labels).float()   # Compare predictions with true labels
    acc = correct.sum() / len(correct)    # Calculate accuracy
    return acc.item()

def train(num_epochs=2000, data_path = "./data/train_combined.csv", patience=4, save_path="./best_model4.pth"):
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    # Read dataframe and map genres to class numbers
    data = pd.read_csv(data_path)
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
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    texts_train = train_data["combined_text"].reset_index(drop=True)
    labels_train = train_data["genre_numbers"].reset_index(drop=True)
    train_dataset = MovieGenreDataset(texts_train, labels_train, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    texts_val = val_data["combined_text"].reset_index(drop=True)
    labels_val = val_data["genre_numbers"].reset_index(drop=True)
    val_dataset = MovieGenreDataset(texts_val, labels_val, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    texts_test = test_data["combined_text"].reset_index(drop=True)
    labels_test = test_data["genre_numbers"].reset_index(drop=True)
    test_dataset = MovieGenreDataset(texts_test, labels_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize training related
    model = GenrePredictor(num_genres=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.2)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    wandb.init(
        project="nlp_movie_genres",
    )

    scaler = torch.amp.GradScaler("cuda")

    # Train loop
    for epoch in range(num_epochs):

        #if epoch == 25:
        #    for param in model.bert.parameters():
        #        param.requires_grad = True
        #
        #        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
        #        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiffzer, mode='min', patience=1, factor=0.1)

        train_loss = 0
        train_correct = 0
        total_train_examples = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #loss.backward()
            #optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy for this batch
            accuracy = calculate_accuracy(logits, labels)
            train_correct += accuracy * len(labels)
            total_train_examples += len(labels)

            wandb.log({"train_batch_loss": loss.item()})
        
        train_loss /= len(train_dataloader)
        train_accuracy = train_correct / total_train_examples

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val_examples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(logits, labels)
                val_loss += loss.item()

                # Calculate validation accuracy for this batch
                accuracy = calculate_accuracy(logits, labels)
                val_correct += accuracy * len(labels)
                total_val_examples += len(labels)

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / total_val_examples

        #scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            "val_loss": val_loss, 
            "val_accuracy": val_accuracy
        })
        print(f"epoch: {epoch} | train loss: {train_loss:.4f} | train acc: {train_accuracy:.4f} | val loss: {val_loss:.4f} | val acc: {val_accuracy:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    # Load the best model for testing
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Test loop
    test_loss = 0
    test_correct = 0
    total_test_examples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            test_loss += loss.item()

            # Calculate test accuracy for this batch
            accuracy = calculate_accuracy(logits, labels)
            test_correct += accuracy * len(labels)
            total_test_examples += len(labels)

    test_loss /= len(test_dataloader)
    test_accuracy = test_correct / total_test_examples

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")

    wandb.log({"test acc": test_accuracy})

    wandb.finish()

def test_model(test_data_path="./data/test_combined.csv", model_path="best_model.pth", out_file_path="results.txt"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = pd.read_csv(data_path)

    test_dataset = MovieGenreTestDataset(data["combined_texts"], tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    
    model = GenrePredictor(num_genres=9).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            genre_idx = torch.argmax(x, dim=1)
            # Map the numbers 0 to 8 to genres
            genres = ["drama", "comedy", "horror", "action", "romance", "western", "animation", "crime", "sci-fi"]

            # Create a list to store the mapped genres
            mapped_genres = [genres[i] for i in genre_idx]

            # Write the mapped genres to a txt file, separated by new lines
            with open(out_file_path, 'w') as f:
                f.write('\n'.join(mapped_genres))

if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    train()