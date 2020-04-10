import torch
from DataLoader import DataLoader
import pdb
import Hyperparameter as param
from Batch import Batch
from Util import numpy_to_var, toData, decoder_initial
from Vocab import Vocab
import numpy as np
import math
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

targetfile1 = "target.txt"
inputfile1 = "ie_out.txt"
device = "cpu"
dataloader1 = DataLoader(inputfile1, targetfile1, device)
x, y = dataloader1.caseload()

vocab_size = 5000
vocab = Vocab(vocab_size)
vocab.w2i = dataloader1.word2idx
print("vocab.w2i")
print(vocab.w2i)
vocab.i2w = dataloader1.idx2word
vocab.count = len(vocab.w2i)
for w in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
    vocab.w2i[w] = vocab.count
    vocab.i2w[vocab.count] = w
    vocab.count += 1
print("<unk>")
print(vocab.w2i['<UNK>'])

targetfile = "test_input.txt"
inputfile = "test_target.txt"
description = ""
target = []
nl = []
with open(targetfile, 'r', encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        if line != "\n":
            description += line

        else:
            target.append(description)
            description = ""
    target.append(description)

with open(inputfile, 'r', encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        if line != "\n":
            description += line
        else:
            nl.append(description)
            description = ""
    nl.append(description)
test = []
new_nl = []
new_target = []
for i in range(min(len(nl), len(target))):
    string = nl[i].split()
    print("string")
    print(string)
    new_nl = [0] * len(string)
    for c in range(len(string)):
        try:
            if string[c] not in vocab.w2i:
                new_nl[c] = "<UNK>"

            else:
                new_nl[c] = vocab.w2i[string[c]]
        except Exception as e:
            # pdb.set_trace()
            print(string[c])
            print(e)

    string = target[i].split()
    new_target = [0] * len(string)
    for c in range(len(string)):
        try:
            if string[c] not in vocab.w2i:
                new_target[c] = "<UNK>"
            else:
                new_target[c] = vocab.w2i[string[c]]
        except Exception as e:
            # pdb.set_trace()
            print(string[c])
            print(e)
    testline = ",".join('%s' %id for id in new_nl) +  "\t" +  ",".join('%s' %id for id in new_target)
    test.append(testline)

print("test")
print(test)

batch = Batch(file_list=[],max_in_len=30,max_out_len=30,max_oovs=12)
batch.num_of_minibatch=len(test)/param.batch_size

# get number of batches
num_samples = len(test)
num_batches = int(num_samples/param.batch_size)


epoch = 10

encoder = torch.load(f='model/encoder_%s_%s.pckl' % ("copynet",str(epoch)))
decoder = torch.load(f='model/decoder_%s_%s.pckl' % ("copynet",str(epoch)))

correct = 0
samples_read = 0
batch.init_batch()
total = len(test)
print_list = []
while(samples_read<len(test)):

    # 1.4.1. initialize gradient buffers
    batch.init_minibatch()

    # 1.4.2. obtain batch outputs
    data = test[samples_read:min(samples_read+param.batch_size,len(test))]
    try:
        inputs, outputs = batch.process_minibatch(data, vocab)
        #inputs, outputs, in_len, out_len = toData(data)
        print("inpurs:")
        print(inputs)
        print("outputs:")
        print(outputs)
    except Exception as e:
        print(e)
        pdb.set_trace()
    samples_read+=len(data)

    # 1.4.3. inputs and outputs must be unk-ed to put into model w/ limited vocab
    unked_inputs = batch.unk_minibatch(inputs,vocab)
    unked_outputs = batch.unk_minibatch(outputs,vocab)

    x = numpy_to_var(unked_inputs)
    print("x")
    print(x.size())
    y = numpy_to_var(unked_outputs)
    print("y")
    print(y.size())

    # 1.5. encoded outputs
    encoded, _ = encoder(x)

    # 1.6.1. get initial input of decoder
    decoder_in, s, w = decoder_initial(x.size(0))
    decoder_in = y[:,0]

    # 1.7. for each decoder timestep
    for j in range(y.size(1) - 1):  # for all sequences
        """
		decoder_in (Variable): [b]
		encoded (Variable): [b x seq x hid]
		input_out (np.array): [b x seq]
		s (Variable): [b x hid]
		"""
        # 1.7.1.1st state - create [out]
        if j == 0:
            out, s, w = decoder(input_idx=y[:, j], encoded=encoded,
                                encoded_idx=inputs, prev_state=s,
                                weighted=w, order=j)
            #             out[2,0,vocab.w2i['codeMirror']]=1
        # remaining states - add results to [out]
        else:
            try:
                tmp_out, s, w = decoder(input_idx=decoder_in.squeeze(), encoded=encoded,
                                    encoded_idx=inputs, prev_state=s,
                                    weighted=w, order=j)
            except Exception as e:
                print(e)
                pdb.set_trace()
            out = torch.cat([out, tmp_out], dim=1)
            print("out")
            print(out)
            print(out.size())
            print("y,size")
            print(y.size())
        # for debugging: stop if nan
        if math.isnan(w[-1][0][0].data[0]):
            print("NaN detected!")
            sys.exit()

            # 1.8.1. select next input
            #         decoder_in = y[:,j] # train with ground truth
        if j == 0:
            try:
                out[0, -1, vocab.w2i['(']] = 1
            except Exception as e:
                print(e)
                pdb.set_trace()
        decoder_in = out[:, -1, :].max(1)[1]  # train with prev outputs
        unked_decoder_in = batch.unk_minibatch(decoder_in.cpu().data.numpy(), vocab)
        #unked_decoder_in = Variable(torch.LongTensor(unked_decoder_in).cuda())
        unked_decoder_in = Variable(torch.LongTensor(unked_decoder_in))
        unk_decoder_in = Variable(torch.LongTensor(decoder_in))
    # 1.9.1. our targeted outputs should include OOV indices
    target_outputs = numpy_to_var(outputs[:, 1:])

    # 1.9.2. get padded versions of target and output
    try:
        target = pack_padded_sequence(target_outputs, batch.output_lens.tolist(), batch_first=True)[0]
    except Exception as e:
        print(e)
        pdb.set_trace()
    pad_out = pack_padded_sequence(out, batch.output_lens.tolist(), batch_first=True)[0]
    for idx in range(len(data)):
        input_print = []
        truth_print = []
        predict_print = []
        for i in inputs[idx]:
            if i == 0:
                break
            else:
                input_print.append(i)
        for i in outputs[idx]:
            if i == 3:
                break
            elif i == 2:
                pass
            else:
                truth_print.append(i)
        for i in out[idx, :, :].max(1)[1].cpu().data.numpy():
            if i == 3:
                break
            else:
                predict_print.append(i)
        line0 = "\n==================================================================="
        print("input_print:")
        print(input_print)
        print("truth_print:")
        print(truth_print)
        try:
            line1 = 'Input1:       ' + ''.join(vocab.idx_list_to_word_list(input_print, batch.idx2oov_list[idx]))
        except Exception as e:
            print(e)
            pdb.set_trace()
        line2 = 'Output:       ' + ''.join(vocab.idx_list_to_word_list(truth_print, batch.idx2oov_list[idx]))
        line3 = 'Predict[UNK]: ' + ''.join(vocab.idx_list_to_word_list(predict_print))
        line4 = 'Predicted:    ' + ''.join(vocab.idx_list_to_word_list(predict_print, batch.idx2oov_list[idx]))
        line1 = line1.replace('var', 'var ')
        line1 = line1.replace(';', ';\nInput2:       ')
        line2 = line2.replace('var', 'var ')
        line3 = line3.replace('var', 'var ')
        line4 = line4.replace('var', 'var ')
        if line2[14:] == line4[14:]:
            correct += 1
            line4 += '\n***CORRECT***'
        print_list.extend([line0, line1, line2, line3, line4])
        print("print_list:")
        print(print_list)
        # with open('test_results_%s_epoch_%d_acc_%1.3f.txt'
        #           %(version,epoch+continue_from,correct*1.0/total),'w') as f:
        #     f.write('\n'.join(print_list))
print(correct * 1.0 / total)
