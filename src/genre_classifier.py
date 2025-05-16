# genre_classifier.py
import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class GenrePredictor:
    def __init__(self, model_path, mlb_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load components with error handling
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.mlb = self._load_mlb(mlb_path)
            self.model = self._load_model(model_path, len(self.mlb.classes_))
            self.MAX_LENGTH = 128
        except Exception as e:
            raise RuntimeError(f"Failed to initialize genre classifier: {str(e)}")

    def _load_mlb(self, mlb_path):
        """Load the MultiLabelBinarizer with error handling"""
        try:
            with open(mlb_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load label binarizer: {str(e)}")

    def _load_model(self, model_path, n_classes):
        """Load the PyTorch model with error handling"""
        try:
            model = GenreClassifier(n_classes)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict_genres(self, overview, threshold=0.5):
        """Make predictions with error handling"""
        try:
            encoding = self.tokenizer.encode_plus(
                overview,
                add_special_tokens=True,
                max_length=self.MAX_LENGTH,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions = (probs > threshold).astype(int).reshape(1, -1)
            genres = self.mlb.inverse_transform(predictions)[0]

            genre_confidences = {
                self.mlb.classes_[i]: float(probs[i])
                for i in range(len(probs)) if predictions[0][i]
            }

            return list(genres), genre_confidences
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")


class GenreClassifier(nn.Module):
    def __init__(self, n_classes, model_name='distilbert-base-uncased'):
        super(GenreClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.out(output)