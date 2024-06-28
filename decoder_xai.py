import torch
from lime.lime_text import LimeTextExplainer
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ModelWrapper:
    def __init__(self, model, vocab, word_to_int, int_to_word, device):
        self.model = model
        self.vocab = vocab
        self.word_to_int = word_to_int
        self.int_to_word = int_to_word
        self.device = device

    def predict(self, texts):
        self.model.eval()
        max_len = 0

        sequences = [[self.word_to_int.get(word, 0) for word in text.split()] for text in texts]

        max_len = max(len(seq) for seq in sequences)

        padded_seqs = [seq + [self.word_to_int.get('<PAD>', 0)] * (max_len - len(seq)) for seq in sequences]
        tensor_seqs = torch.LongTensor(padded_seqs).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor_seqs)

        probs = torch.softmax(predictions, dim=-1).cpu().numpy()
        probs = probs[:, -1, :]

        return probs


def explain_prediction(text, num_features=10):
    exp = explainer.explain_instance(text, model_wrapper.predict, num_features=num_features)
    exp.show_in_notebook(text=True)
