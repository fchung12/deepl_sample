import os
import sys
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.nn.utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import rouge

# Constants
EMBEDDING_SIZE = 30522
LABEL_SIZE = 100
SUMMARY_MAX_LENGTH = 48 # Max sum
TEXT_MAX_LENGTH = 2018 # Max length of the input
TEXT_CUTOFF_LENGTH = 2000 # Cutoff length
filelimit = 1000
# Hyperparameters
num_epochs = 250
num_epochs_test = 5 # Testing
batch_size = 8 # 20 is too high
hidden_dim = 1000
discriminator_traink = 3
generator_traink = 1
overlap = 10 # BERT Overlap
learning_rate_g = 1e-3 # old 0.01
learning_rate_d = 1e-4

# Assuming that the data directory is provided as sys.argv[1]
data_dir_train = sys.argv[1]
data_dir_val = sys.argv[2]
data_dir_test = sys.argv[3]

#https://github.com/huggingface/pytorch-pretrained-BERT#usage
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer
# Load pre-trained model (weights)
model_version = 'bert-base-uncased'
bert_model = BertForMaskedLM.from_pretrained(model_version)
bert_model.eval()
cuda = torch.cuda.is_available()
if cuda:
    bert_model = bert_model.cuda()
    
# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained(model_version)    
    
#pad missing length with "." for cuda matrix
def padtolength(lst, length):
    if (len(lst) % (length)) != 0:
        lst += (['[PAD]'] * (length - (len(lst)%(length))))
    return lst
        
#Splits string into equal length pieces with overlap
def to512(tokenlist, trunclen ,overlap):
    returnlist = []
    #loops throuch and slices list
    i = 0
    while(i < len(tokenlist)):
        if i == 0:
            returnlist.append(padtolength(tokenlist[i:(i+trunclen)], trunclen))
            i = i + trunclen
        else:
            returnlist.append(padtolength(tokenlist[i-overlap:(i-overlap+trunclen)],trunclen))
            i = i + trunclen - overlap
    return(returnlist)
  
#Removes BERT overlap in tensor, deletes first (overlap) elements and reshapes
def remove_overlap(input_tensor, summ = False): 
    np_tensor = input_tensor.cpu().numpy()
    mask = np.ones(np_tensor.shape)
    for i in range (1,mask.shape[0]):
        mask[i][0:overlap] = 0
    mask = mask.astype(bool)
    if summ == False:
        deleted = np_tensor[mask].reshape(512*(mask.shape[0]) - overlap*(mask.shape[0]-1) , 30522)
    else:
        deleted = np_tensor[mask].reshape(48*(mask.shape[0]) - overlap*(mask.shape[0]-1) , 30522)
    deleted = torch.from_numpy(deleted).cuda()
    return(deleted)

#Pads or truncates text tokens to max_length
def pad_text_to_length(text, max_length):
    if (len(text) < max_length):
        return text + ['[PAD]'] * (max_length - len(text))
    elif (len(text) > max_length):
        return text[0:max_length]
    else:
        return text

class TextDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        return

    def __len__ (self):

        totallen = 0
        for subdir in os.listdir(self.input_dir):
           totallen += len(os.listdir(os.path.join(self.input_dir,subdir)))
        return totallen
      
    # loads in input image and mask for each index on the dataset
    def __getitem__(self,idx):
        #doc_path = os.path.join(self.input_dir, str(idx))
        superpath = os.path.join(self.input_dir, str(int(int(idx)/int(filelimit))))
        doc_path = os.path.join(superpath, str(idx % filelimit))
        
        text_file = os.path.join(doc_path,"text.txt")
        summ_file = os.path.join(doc_path,"summ.txt")
        
        text = open(text_file).read()
        summ = open(summ_file).read()
 
        text_tokens = bert_tokenizer.tokenize(text)
        summ_tokens = bert_tokenizer.tokenize(summ)       
    
        text_tokens = pad_text_to_length(text_tokens, TEXT_CUTOFF_LENGTH)
        summ_tokens = pad_text_to_length(summ_tokens, SUMMARY_MAX_LENGTH)
                 
        text_tokens_512 = to512(text_tokens,512,overlap)
        summ_tokens_512 = to512(summ_tokens,48,overlap)
        
        text_toids = []
        for list_512 in text_tokens_512:
            text_toids.append(bert_tokenizer.convert_tokens_to_ids(list_512))
        text_tensor = torch.tensor(text_toids)
        text_tensor = text_tensor.cuda()
        
        summ_toids = []
        for list_512 in summ_tokens_512:
            summ_toids.append(bert_tokenizer.convert_tokens_to_ids(list_512))
        summ_tensor = torch.tensor(summ_toids)
        summ_tensor = summ_tensor.cuda()
        
        with torch.no_grad():
            return remove_overlap(bert_model(text_tensor)), remove_overlap(bert_model(summ_tensor), summ = True)

cornell_newsroom_train = TextDataset(input_dir = data_dir_train)
cornell_newsroom_val = TextDataset(input_dir = data_dir_val)
cornell_newsroom_test = TextDataset(input_dir = data_dir_test)

rand_sampler_train = torch.utils.data.RandomSampler(cornell_newsroom_train, replacement=False)
rand_sampler_val = torch.utils.data.RandomSampler(cornell_newsroom_val, replacement=False)
rand_sampler_test = torch.utils.data.RandomSampler(cornell_newsroom_test, replacement=False)
train_sampler = torch.utils.data.DataLoader(cornell_newsroom_train, batch_size=batch_size, sampler=rand_sampler_train)
val_sampler = torch.utils.data.DataLoader(cornell_newsroom_val, batch_size=batch_size, sampler=rand_sampler_val)
test_sampler = torch.utils.data.DataLoader(cornell_newsroom_test, batch_size=batch_size, sampler=rand_sampler_test)

def batch(val = False, test = False):
    if val:
      iterator = iter(val_sampler)
    elif test:
      iterator = iter(test_sampler)
    else:
      iterator = iter(train_sampler)
    
    # Contains (text batch tensors, summ batch tensors)
    # ((10,2018,30522),(10,512,30522))
    return(iterator.next())

class Discriminator(nn.Module):
  def __init__(self, dropout_p = 0.3):
    super(Discriminator, self).__init__()
    
    self.summ_conv2 = nn.Sequential(
      nn.Conv2d(1,100,(2,EMBEDDING_SIZE)), # 20 x 47 x 1
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d((47,1))
    )
    
    self.summ_conv3 = nn.Sequential(
      nn.Conv2d(1,100,(3,EMBEDDING_SIZE)), # 20 x 46 x 1
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d((46,1))
    )
    
        
    self.summ_conv4 = nn.Sequential(
      nn.Conv2d(1,100,(4,EMBEDDING_SIZE)), # 20 x 45 x 1
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d((45,1))
    )
    
        
    self.summ_conv5 = nn.Sequential(
      nn.Conv2d(1,100,(5,EMBEDDING_SIZE)), # 20 x 44 x 1
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d((44,1))
    )
    
    '''
    self.text_conv2 = nn.Sequential(
      nn.Conv2d(1,100,(2,EMBEDDING_SIZE)), # 20 x 2017 x 1
      nn.ReLU(),
      nn.MaxPool2d((2017,1))
    )
    
    self.text_conv3 = nn.Sequential(
      nn.Conv2d(1,100,(3,EMBEDDING_SIZE)), # 20 x 2016 x 1
      nn.ReLU(),
      nn.MaxPool2d((2016,1))
    )
    
    self.text_conv4 = nn.Sequential(
      nn.Conv2d(1,100,(4,EMBEDDING_SIZE)), # 20 x 2015 x 1
      nn.ReLU(),
      nn.MaxPool2d((2015,1))
    )
      
    self.text_conv5 = nn.Sequential(
      nn.Conv2d(1,100,(5,EMBEDDING_SIZE)), # 20 x 2014 x 1
      nn.ReLU(),
      nn.MaxPool2d((2014,1))
    )
    '''
    self.summ_linear = nn.Linear(400,2)
    self.text_linear = nn.Linear(400,30)
    
    self.intermediate_linear = nn.Linear(400, 400)
    #self.final_linear = nn.Linear(60,2)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(1)
   
    self.sigmoid = nn.Sigmoid()
    self.dropout_p = dropout_p
    self.dropout = nn.Dropout(self.dropout_p)
  
    
    
  # NOTE: Conv2D expacts Batch x Channel x Height x Width
  def forward(self, input, text, printB = False, final_layer = True):
    text = text.unsqueeze(1)
    input = input.unsqueeze(1)

    summ_conv2 = self.summ_conv2(input).view(-1, 100)
    summ_conv3 = self.summ_conv3(input).view(-1, 100)
    summ_conv4 = self.summ_conv4(input).view(-1, 100)
    summ_conv5 = self.summ_conv5(input).view(-1, 100)
 
    '''
    text_conv2 = self.text_conv2(text).view(-1, 100)
    text_conv3 = self.text_conv3(text).view(-1, 100)
    text_conv4 = self.text_conv4(text).view(-1, 100)
    text_conv5 = self.text_conv5(text).view(-1, 100)
    '''
    summ_linear_input = torch.cat((summ_conv2, summ_conv3, summ_conv4, summ_conv5), dim=1)
    #text_linear_input = torch.cat((text_conv2, text_conv3, text_conv4, text_conv5), dim=1)
    summ_linear_input = self.dropout(summ_linear_input)
    summ_linear_output1 = self.relu(self.intermediate_linear(summ_linear_input))
    summ_linear_output1 = self.dropout(summ_linear_output1)
    #summ_linear_output = self.relu(summ_linear_output)
    '''
    text_linear_output = self.text_linear(text_linear_input)
    text_linear_output = self.relu(text_linear_output)
    
    summ_text_input = torch.cat((summ_linear_output, text_linear_output), dim = 1)
    summ_text_output = self.final_linear(summ_text_input)
    '''
      #print(text_linear_output)
      #print(summ_text_output)
  
    #return self.softmax(summ_linear_output)
    if final_layer:
        summ_linear_output2 = self.summ_linear(summ_linear_output1)
        if printB:
            print(summ_linear_output2)
        return summ_linear_output2
    else:
        return summ_linear_output1
    #return self.softmax(summ_text_output)
    
    '''
    text_conv = self.conv_text(text)
    text_conv = text_conv.view(-1, 40*125)
    text = self.text_linear(text_conv)
    summ_conv = self.conv_summ(input)
    
    summ_conv = summ_conv.view(-1, 250*3)

    linear_input = torch.cat((text, summ_conv), dim=1)
    '''
    
    #return self.sigmoid(self.summ_linear(linear_input))
    #return self.sigmoid(linear_input[:][0])

class Generator(nn.Module):
  def __init__(self, dropout_p=0.5):
    super(Generator, self).__init__()
    
    self.lstm_enc = nn.LSTM(EMBEDDING_SIZE, hidden_dim)
    self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim)
    
    self.dropout_p = dropout_p
    self.embedding_dec = nn.Linear(EMBEDDING_SIZE+10000, hidden_dim)
    self.dropout = nn.Dropout(self.dropout_p)
    
    self.attn = nn.Linear(hidden_dim * 3, TEXT_MAX_LENGTH)
    self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
    
    self.out = nn.Linear(hidden_dim, EMBEDDING_SIZE)
  
  # LSTM expectes SeqLen (TextMaxLength) x Batch x Features
  def forward(self, input, target_size, teacher_forcing = 0.25, summaries = None):
    input = torch.transpose(input, 0, 1)
    # Encoder
    # enc_output: SeqLen x Batch x Hidden
    # hidden[0]: 1 x Batch x Hidden
    enc_output, hidden = self.lstm_enc(input)
    #print(enc_output.shape)
    #print(len(hidden))
    
    #input = torch.tensor([[0] * batch_size * EMBEDDING_SIZE]).view(batch_size, -1).float().cuda()
    input = torch.from_numpy(np.random.randn(batch_size, EMBEDDING_SIZE)).float().cuda()
    
    summary = torch.zeros(target_size, batch_size, EMBEDDING_SIZE).cuda()
    
    for i in range(target_size):
    # Decoder (Attention)
      noise = torch.from_numpy(np.random.randn(batch_size,10000)).float().cuda()
      input = torch.cat((input,noise),dim = 1)
      embedded = self.embedding_dec(input).view(1, batch_size, -1)
      embedded = self.dropout(embedded)
      # embedded[0]: 1 x Batch x Hidden

      noise2 = torch.from_numpy(np.random.randn(1, batch_size, hidden_dim)).float().cuda()
      attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0],noise2), 2).squeeze(0)), dim=1)
      
      # attn_weights: Batch x SeqLen

      # B x SeqLen matmul 
      attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_output.transpose(0, 1)).transpose(0, 1)

      
      output = torch.cat((embedded[0].squeeze(0), attn_applied[0]), 1) # B x H cat B x H
      output = self.attn_combine(output) # Take the concatenated components and merge them
      #B x H
      output = F.relu(output)
      
      output, hidden = self.lstm_dec(output.unsqueeze(0), hidden)
      input = self.out(output[0])
      summary[i] = input

      if summaries is not None and np.random.random() < teacher_forcing:
          input = summaries[:,i]
     
    return summary

# Given a SINGLE example (e.g. preds[0]), converts to human readable format
def embedToString(example):
    pred_idxs = torch.argmax(example, dim = 1).tolist()
    pred_toks = bert_tokenizer.convert_ids_to_tokens(pred_idxs)
    #tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##ed', 'wa', 'un', 'runn', '##ing', ',']
    text = ' '.join([x for x in pred_toks])
    fine_text = text.replace(' ##', '')
    return fine_text

def train_model():
  print("Training Model")
  #noise_size = 64

  generator = Generator().cuda()
  discriminator = Discriminator().cuda()

  g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate_g)
  d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)

  criterion = nn.CrossEntropyLoss()
  l2loss = nn.MSELoss()

  g_losses = []
  d_losses = []
  g_val_losses = []
  d_val_losses = []

  #true_one_hot = (torch.from_numpy(np.array([[1,0] * batch_size]))).view(-1, 2).float()
  #fake_one_hot = (torch.from_numpy(np.array([[0,1] * batch_size]))).view(-1, 2).float()

  for epoch in range(num_epochs):
    print("Epoch " + str(epoch+1))
    generator.train()
    discriminator.train()
    # Training Discriminator
    avg_loss = 0
    avg_accuracy = 0
    accuracy_fake = 0
    accuracy_real = 0
    for i in range(discriminator_traink):
      sequences, real_summaries = batch()
  
      sequences = sequences.cuda()
      real_summaries = real_summaries.cuda()

      generated_summaries = generator(sequences, SUMMARY_MAX_LENGTH).transpose(0, 1) # CNN needs Batch to be first, not second
      
      #print("Generated Summaries - Shape: " + str(generated_summaries.shape)) # Batch x SumLen x Emb
      #print("Real Summaries - Shape: " + str(real_summaries.shape)) # Batch x SumLen x Emb
      true_out = discriminator(real_summaries, sequences, printB = False)
      fake_out = discriminator(generated_summaries, sequences, printB = False)

      #print(true_out)
      #print(torch.sum(true_out[:,0] >= 0.5).float()/(torch.sum(true_out[:,0] >= 0)))
      #print(fake_out)
      #print(torch.sum(fake_out[:,0] < 0.5).float()/(torch.sum(fake_out[:,0] >=0)))
      #accuracy_real += torch.sum(true_out[:,0] >= 0.5).float()/(torch.sum(true_out[:,0] >= 0))/discriminator_traink
      #accuracy_fake += torch.sum(fake_out[:,0] < 0.5).float()/(torch.sum(fake_out[:,0] >=0))/discriminator_traink
      true_out_s = F.softmax(true_out, 1)
      fake_out_s = F.softmax(fake_out, 1)
      accuracy_real += torch.sum(true_out_s[:,0] >= 0.5).float()/(torch.sum(true_out_s[:,0] >= 0))/discriminator_traink
      accuracy_fake += torch.sum(fake_out_s[:,0] < 0.5).float()/(torch.sum(fake_out_s[:,0] >=0))/discriminator_traink
      
      #real_loss = criterion(true_out, torch.ones(true_out.shape).cuda())
      #fake_loss = criterion(fake_out, torch.zeros(fake_out.shape).cuda()) 
      distribution = np.random.uniform(0, 1, batch_size)
      distribution[distribution >= 0.9] = 1
      distribution[distribution < 0.9] = 0
      true_one_hot = (torch.from_numpy(distribution)).view(-1).long()
      distribution = np.random.uniform(0, 1, batch_size)
      distribution[distribution >= 0.9] = 0
      distribution[distribution < 0.9] = 1
      fake_one_hot = (torch.from_numpy(distribution)).view(-1).long()

      real_loss = criterion(true_out, true_one_hot.cuda())
      fake_loss = criterion(fake_out, fake_one_hot.cuda())
      
      loss = real_loss + fake_loss
      
      d_optimizer.zero_grad()
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(discriminator.parameters(),5)

      d_optimizer.step()
      
      #avg_loss += loss.cpu().detach().numpy()
      avg_loss += loss.item()
      sys.stdout.flush()

      
    print("Real Accuracy " + str(accuracy_real))
    print("Fake Accuracy " + str(accuracy_fake))    
      
    d_losses.append(avg_loss / discriminator_traink)

    g_avg_loss = 0.0
    fine_text = ""

    true_one_hot = (torch.from_numpy(np.array([[0] * batch_size]))).view(-1)
    fake_one_hot = (torch.from_numpy(np.array([[1] * batch_size]))).view(-1)
    for j in range(generator_traink):

        # Training Generator
        sequences, summaries = batch()
        sequences = sequences.cuda()
        summaries = summaries.cuda()

        generated_summaries = generator(sequences, SUMMARY_MAX_LENGTH, summaries = summaries).transpose(0, 1) # CNN needs Batch to be first, not second
        fake_out = discriminator(generated_summaries, sequences, final_layer = False)
    
        #g_loss = criterion(fake_out, torch.ones(fake_out.shape).cuda())
        #print("DISCRIMINATOR_OUTPUT:" + str(fake_out))

        #g_loss = criterion(fake_out, true_one_hot.cuda())
        real_out = discriminator(summaries, sequences, final_layer = False)
        g_loss = l2loss(fake_out, real_out)
   
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        #g_avg_loss += g_loss.cpu().detach().numpy() / generator_traink
        g_avg_loss += g_loss.item()/generator_traink

    print("Example Summary: ")
    preds = generated_summaries
    fine_text = embedToString(preds[0])     
    print(fine_text)

    g_losses.append(g_avg_loss)

    avg_g_loss = 0.0
    avg_d_loss = 0.0

    # Validation
    if epoch % 10 == 0:
        generator.eval()
        discriminator.eval()
        for j in range(10):
            # Training Generator
            sequences, summaries = batch(val = True)
            sequences = sequences.cuda()
            summaries = summaries.cuda()
    
            generated_summaries = generator(sequences, SUMMARY_MAX_LENGTH).transpose(0, 1) # CNN needs Batch to be first, not second

            fake_out = discriminator(generated_summaries, sequences, final_layer = False)
            real_out = discriminator(summaries, sequences, final_layer = False)
            g_loss = l2loss(fake_out, real_out)

            fake_out = discriminator(generated_summaries, sequences, final_layer = False)
            true_out = discriminator(summaries, sequences, final_layer = True)
            real_loss = criterion(true_out, true_one_hot.cuda())
            fake_loss = criterion(fake_out, fake_one_hot.cuda())
            d_loss = real_loss + fake_loss
       
            avg_g_loss += g_loss.item()/10
            avg_d_loss += d_loss.item()/10

        g_val_losses.append(avg_g_loss)
        d_val_losses.append(avg_d_loss)
        print("Validation: generator val loss = " + str(g_val_losses[-1]) + ", discriminator val loss = " + str(d_val_losses[-1]))

    print("Train: generator loss = " + str(g_losses[-1]) + ", discriminator loss = " + str(d_losses[-1]))

    fig_tr, ax_tr = plt.subplots()
    ax_tr.plot(d_losses, label="Discriminator (Train)")
    ax_tr.plot(g_losses, label="Generator (Train)")
    #ax.plot(g_losses, d_losses, g_val_losses, d_val_losses)
    ax_tr.legend()
    #plt.show()
    fig_tr.savefig("TrainValLosses.png")

    # Save Model Parameters to file
    # Runs every epoch so that the training can be terminated at any time and still maintain weights
    print("Updating model weights to file...")
    torch.save(generator.state_dict(), "generator_weights.torch")
    torch.save(discriminator.state_dict(), "discriminator_weights.torch")

def test_model():
    print("Testing Model")

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate_g)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)
    
    criterion = nn.CrossEntropyLoss()
    l2loss = nn.MSELoss()
    
    g_test_losses = []
    d_test_losses = []

    real_summaries_txt = []
    gen_summaries_txt = []
    
    generator.eval()
    discriminator.eval()

    checkpoint_gen = torch.load("generator_weights.torch")
    checkpoint_dec = torch.load("discriminator_weights.torch")
    generator.load_state_dict(checkpoint_gen)
    discriminator.load_state_dict(checkpoint_dec)

    true_one_hot = (torch.from_numpy(np.array([[0] * batch_size]))).view(-1)
    fake_one_hot = (torch.from_numpy(np.array([[1] * batch_size]))).view(-1)
    
    for epoch in range(num_epochs_test):
        print("Epoch " + str(epoch+1))
        avg_g_loss = 0.0
        avg_d_loss = 0.0
        sequences, summaries = batch(val = True)
        sequences = sequences.cuda()
        summaries = summaries.cuda()
        
        generated_summaries = generator(sequences, SUMMARY_MAX_LENGTH).transpose(0, 1) # CNN needs Batch to be first, not second

        # Store generated summaries for ROUGE
        for i in range(batch_size):
            gen_summaries_txt.append(embedToString(generated_summaries[i]))
            real_summaries_txt.append(embedToString(summaries[i]))
        
        fake_out = discriminator(generated_summaries, sequences, final_layer = False)
        real_out = discriminator(summaries, sequences, final_layer = False)
        g_loss = l2loss(fake_out, real_out)
        
        fake_out = discriminator(generated_summaries, sequences, final_layer = False)
        true_out = discriminator(summaries, sequences, final_layer = True)
        real_loss = criterion(true_out, true_one_hot.cuda())
        fake_loss = criterion(fake_out, fake_one_hot.cuda())
        d_loss = real_loss + fake_loss
        
        avg_g_loss += g_loss.item()/generator_traink
        avg_d_loss += d_loss.item()/generator_traink
        
        g_test_losses.append(avg_g_loss)
        d_test_losses.append(avg_d_loss)
        print("Test: generator test loss = " + str(g_test_losses[-1]) + ", discriminator test loss = " + str(d_test_losses[-1]))

    # Rouge Report
    evaluator = rouge.Rouge()
    rouge_scores = evaluator.get_scores(gen_summaries_txt, real_summaries_txt)
    rouge_l_all_p = []
    rouge_1_all_p = []
    rouge_2_all_p = []
    rouge_l_all_r = []
    rouge_1_all_r = []
    rouge_2_all_r = []

    for entry in rouge_scores:
        rouge_l_all_p.append(entry["rouge-l"]["p"])
        rouge_1_all_p.append(entry["rouge-1"]["p"])
        rouge_2_all_p.append(entry["rouge-2"]["p"])
        rouge_l_all_r.append(entry["rouge-l"]["r"])
        rouge_1_all_r.append(entry["rouge-1"]["r"])
        rouge_2_all_r.append(entry["rouge-2"]["r"])

    avg_l_p = np.asarray(rouge_l_all_p).mean()
    avg_1_p = np.asarray(rouge_1_all_p).mean()
    avg_2_p = np.asarray(rouge_2_all_p).mean()
    avg_l_r = np.asarray(rouge_l_all_r).mean()
    avg_1_r = np.asarray(rouge_1_all_r).mean()
    avg_2_r = np.asarray(rouge_2_all_r).mean()

    print("Rouge-L Precision; Recall: " + str(avg_l_p) + "; " + str(avg_l_r))
    print("Rouge-1 Precision; Recall: " + str(avg_1_p) + "; " + str(avg_1_r))
    print("Rouge-2 Precision; Recall: " + str(avg_2_p) + "; " + str(avg_2_r))
    
# Train the model
train_model()

# Test the model
test_model()
