from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import GenrePredictor
from data import Dataset
from tqdm import tqdm
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")  # For Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # For NVIDIA GPUs
else:
    device = torch.device("cpu")    # Fallback to CPU

def train(num_epochs=100):
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Sample data
    texts = ["A great movie about friendship.", "An action-packed thriller."]
    labels = [0, 1] 

    # Create the dataset
    dataset = Dataset(texts, labels, tokenizer)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize the model
    model = GenrePredictor(num_genres=9) 

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    loss_fn = torch.nn.CrossEntropyLoss()


    # Example forward pass (for testing)
    for epoch in tqdm(range(num_epochs)):
        for batch in dataloader:

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch}, loss: {loss.backward:.4f}")

if __name__ == "__main__":
    train()