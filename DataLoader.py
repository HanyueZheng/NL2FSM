import torch
from torch.autograd import Variable
class DataLoader():
    def __init__(self, filename, targetfile, device):
        self.filename = filename
        self.targetfile = targetfile
        self.device = device
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def caseload(self):
        nl = []
        target = []
        description = ""
        tokens = 0
        with open(self.targetfile, 'r', encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if  line != "\n":
                    description += line
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        if not word in self.word2idx:
                            self.word2idx[word] = self.idx
                            self.idx2word[self.idx] = word
                            self.idx += 1

                else:
                    target.append(description)
                    description = ""
            target.append(description)

        with open(self.filename, 'r', encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if  line != "\n":
                    description += line
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        if not word in self.word2idx:
                            self.word2idx[word] = self.idx
                            self.idx2word[self.idx] = word
                            self.idx += 1
                else:
                    nl.append(description)
                    description = ""
            nl.append(description)
        return nl, target

    def list_string(self, string):
        string = string.split()
        list = [0] * len(string)
        #tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                list[c] = self.word2idx[string[c]]
            except Exception as e:
                #pdb.set_trace()
                print(string[c])
                print(e)
        return list
