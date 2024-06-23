from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from utils import log_execution_time, logger

class EmbeddingsGenerator:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @log_execution_time
    def generate_embeddings(self, texts):
        self.model.eval()
        embeddings = []
        with torch.no_grad():  # Disable gradient calculation
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                outputs = self.model(**inputs)
                # Use mean pooling of the token embeddings to get a single embedding per text
                embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)  # Stack the list of arrays into a single array