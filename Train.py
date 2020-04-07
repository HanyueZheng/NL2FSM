import Hyperparameter as param
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import Decoder

criterion = nn.NLLLoss()
start = time.time()
decoder = Decoder(vocab_size, embed_size, hidden_size)
for epoch in range(param.num_epochs):
    print("==================================================")
    print("Epoch ",epoch+1)
    opt_d = optim.Adam(params=decoder.parameters(), lr=lr)
    lr= lr * param.weight_decay # weight decay
    # shuffle data
    samples_read = 0
    while(samples_read<len(train)):
        # initialize gradient buffers
        opt_d.zero_grad()

        # obtain batch outputs
        batch = train[samples_read:min(samples_read+batch_size,len(train))]
        input_out, output_out, in_len, out_len = toData(batch)
        samples_read+=len(batch)

        # mask input to remove padding
        input_mask = np.array(input_out>0, dtype=int)

        # input and output in Variable form
        x = numpy_to_var(input_out)
        y = numpy_to_var(output_out)

        # apply to encoder
        encoded, _ = encoder(x)

        # get initial input of decoder
        decoder_in, s, w = decoder_initial(x.size(0))

        # out_list to store outputs
        out_list=[]
        for j in range(y.size(1)): # for all sequences
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
            if math.isnan(w[-1][0][0].data[0]):
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
        print("[%d/%d] Loss: %1.4f"%(samples_read,len(train),loss.data[0]))
        opt_d.step()
        step += 1
        info = {
            'loss': loss.data[0]
        }
    # print("Loss: ",loss.data[0])
    elapsed = time.time()
    print("Elapsed time for epoch: ",elapsed-start)
    start = time.time()