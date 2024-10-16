import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

"""
Class containing the model used to predict genres of movies
"""
class GenrePredictor(nn.Module):
    def __init__(self, num_genres = 9) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert = AutoModel.from_pretrained('distilbert/distilbert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(64, num_genres)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        #outputs = self.bert(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    token_type_ids=token_type_ids
        #)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_token_embedding = outputs[0][:, 0, :]  # Shape: [batch_size, hidden_size]

        
        logits = self.classification_head(cls_token_embedding)
        return logits

