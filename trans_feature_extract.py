from transformers import pipeline
import torch

nlp = pipeline('feature-extraction')
filename = "input.txt"
with open(filename, 'r') as f:
    lines = f.readlines()
    outfile = open("feature_out.txt", "w")
    for line in lines:
        print(nlp(line), file=outfile)