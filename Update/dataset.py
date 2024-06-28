import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Custom PyTorch dataset for text data."""

    def __init__(self, samples, word_to_int, pad=True, max_length=None, pad_token='[PAD]'):
        """
        Initialize the TextDataset.
        
        Args:
            samples (list of lists): Each element of samples should be a list of words.
            word_to_int (dict): Dictionary mapping words to integers.
            pad (bool): Whether to pad sequences to `max_length`.
            max_length (int): Maximum length of sequences. Required if `pad` is True.
            pad_token (str): Token used for padding.
        """
        self.samples = samples
        self.word_to_int = word_to_int
        self.pad = pad
        self.max_length = max_length
        self.pad_token = pad_token
        if pad and max_length is None:
            raise ValueError("max_length must be specified if padding is enabled.")
        
        # Ensure PAD token is in the dictionary
        if pad_token not in self.word_to_int:
            self.word_to_int[pad_token] = len(self.word_to_int)
        
    def __len__(self):
        """Get the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing the input sequence and target sequence.
        """
        sample = self.samples[idx]
        if self.pad:
            # Ensure sample is a list of words; pad or truncate the sample
            if len(sample) < self.max_length:
                sample = sample + [self.pad_token] * (self.max_length - len(sample))
            elif len(sample) > self.max_length:
                sample = sample[:self.max_length]

        input_seq = torch.LongTensor([self.word_to_int.get(word, self.word_to_int[self.pad_token]) for word in sample[:-1]])
        target_seq = torch.LongTensor([self.word_to_int.get(word, self.word_to_int[self.pad_token]) for word in sample[1:]])

        return input_seq, target_seq



# VERSION 1
# import torch
# from torch.utils.data import Dataset

# class TextDataset(Dataset):
#     """Custom PyTorch dataset for text data."""

#     def __init__(self, samples, word_to_int):
#         """
#         Initialize the TextDataset.
        
#         Args:
#             samples (list): List of text samples.
#             word_to_int (dict): Dictionary mapping words to integers.
#         """
#         self.samples = samples
#         self.word_to_int = word_to_int

#     def __len__(self):
#         """Get the total number of samples in the dataset."""
#         return len(self.samples)

#     def __getitem__(self, idx):
#         """
#         Get a sample from the dataset at the given index.
        
#         Args:
#             idx (int): Index of the sample to retrieve.
        
#         Returns:
#             tuple: A tuple containing the input sequence and target sequence.
#         """
#         sample = self.samples[idx]
#         input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]])
#         target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]])
#         return input_seq, target_seq
