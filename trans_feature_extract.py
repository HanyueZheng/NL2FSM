import torch
from DataLoader import*
from transformers import *

filename = "ie_out.txt"
targetfile = "target.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(filename, targetfile, device)
dataset, targetset = dataloader.caseload()
