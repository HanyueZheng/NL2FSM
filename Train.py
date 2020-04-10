import Hyperparameter as param
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import Model
from Model import CopyDecoder, CopyEncoder
from DataLoader import DataLoader
import Hyperparameter as param
from Util import numpy_to_var, toData, decoder_initial
import torch
import sys
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

criterion = nn.NLLLoss()
start = time.time()
targetfile = "target.txt"
inputfile = "ie_out.txt"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
dataloader = DataLoader(inputfile, targetfile, device)
x, y = dataloader.caseload()
train = []
for i in range(min(len(x), len(y))):
    try:
        x_new = [str(x) for x in dataloader.list_string(x[i])]
        y_new = [str(y) for y in dataloader.list_string(y[i])]
        trainline = ",".join(x_new) + "\t" +  ",".join(y_new)
    except Exception as e:
        pdb.set_trace()
        print(e)
    train.append(trainline)
vocab_size = 5000

# get number of batches
num_samples = len(train)
num_batches = int(num_samples/param.batch_size)

encoder = CopyEncoder(vocab_size, param.embed_size, param.hidden_size)
decoder = CopyDecoder(vocab_size, param.embed_size, param.hidden_size)
# if torch.cuda.is_available():
#     encoder.cuda()
#     decoder.cuda()

for epoch in range(param.num_epochs):
    print("==================================================")
    print("Epoch ",epoch+1)
    opt_e = optim.Adam(params=encoder.parameters(), lr=param.lr)
    opt_d = optim.Adam(params=decoder.parameters(), lr=param.lr)
    lr= param.lr * param.weight_decay # weight decay
    # shuffle data
    samples_read = 0
    while(samples_read<len(train)):
        # initialize gradient buffers
        opt_d.zero_grad()

        # obtain batch outputs
        batch = train[samples_read:min(samples_read+param.batch_size,len(train))]
        #batch = train[samples_read]
        input_out, output_out, in_len, out_len = toData(batch)
        samples_read+=len(batch)
        #samples_read += 1

        # mask input to remove padding
        input_mask = np.array(input_out>0, dtype=int)

        # input and output in Variable form
        x = numpy_to_var(input_out)
        y = numpy_to_var(output_out)
        x.size()
        y.size()

        # apply to encoder
        encoded, _ = encoder(x)
        encoded.size()

        # get initial input of decoder
        decoder_in, s, w = decoder_initial(x.size(0))
        decoder_in.size()
        # out_list to store outputs
        out_list=[]
        for j in range(y.size(1)):# for all sequences
            y.size()
            """
            decoder_in (Variable): [b]
            encoded (Variable): [b x seq x hid]
            input_out (np.array): [b x seq]
            s (Variable): [b x hid]
            """
            # 1st state
            if j==0:
                out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
            # remaining states
            else:
                tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
                out = torch.cat([out,tmp_out],dim=1)

            # for debugging: stop if nan
            if math.isnan(w[-1][0][0].item()):
                sys.exit()
            # select next input

            if epoch % 2 ==13:
                decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
            else:
                decoder_in = y[:,j] # train with ground truth
            out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())

        # print(torch.stack(decoder.prob_c_to_g,1))
        target = pack_padded_sequence(y,out_len.tolist(), batch_first=True)[0]
        pad_out = pack_padded_sequence(out,out_len.tolist(), batch_first=True)[0]
        # include log computation as we are using log-softmax and NLL
        pad_out = torch.log(pad_out)
        loss = criterion(pad_out, target)
        loss.backward()
        # if samples_read%500==0:
        print("[%d/%d] Loss: %1.4f"%(samples_read,len(train),loss.item()))
        opt_d.step()
        param.step += 1
        info = {
            'loss': loss.item()
        }
    # print("Loss: ",loss.data[0])
    elapsed = time.time()
    print("Elapsed time for epoch: ",elapsed-start)
    start = time.time()

    if epoch % 10 == 0:
        torch.save(f='model/encoder_%s_%s.pckl' % ("copynet", str(epoch)), obj=encoder)
        torch.save(f='model/decoder_%s_%s.pckl' % ("copynet", str(epoch)), obj=decoder)