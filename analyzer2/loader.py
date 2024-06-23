import torch
import string
import emoji

# can also use demoji package
# demoji.download_codes()

class Message():
    def __init__(self, message: str):
        self.message = message

    def extractEmojis(self, message):
        emoji.demojize(message, delimiters("", ""))
        return message

    def letterToTensor(self, letter):
        tensor = torch.zeros(1, n_letters)
        tensor[0][letterToIndex(letter)] = 1
        return tensor

    def messageToTensor(self, message):
        tensor = torch.zeros(len(message), 1, n_letters)
        message = message.split()
        for 

class Dataset():
    pass
