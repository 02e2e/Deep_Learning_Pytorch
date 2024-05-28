



import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

data = [('professor', 'you smell', 'hi there you are fun', 1000, 1), 
        ('professor', 'you fail', 'hi there you are cool', 5000, 1), 
        ('professor', 'youre bad', 'we enjoy your company', 6000, 1),
        ('professor', 'above average', 'i like you', 10000, 0),
        ('professor', 'hate you', 'lets get together', 4000, 1),
        ('professor', 'wiz kid', 'you should be happier', 6000, 0),
        ('professor', 'amazing job', 'we are great together', 700, 0),
        ('brother', 'great job', 'we love you', 500, 1),
        ('brother', 'wiz kid', 'you are the best', 600, 1),
        ('brother', 'you fail', 'you rock', 700, 0),
        ('brother', 'hate you', 'you suck', 600, 0),
        ('brother', 'you smell', 'we do not like you', 1000, 0),
        ('mom', 'you smell', 'we should go further', 2000, 0),
        ('mom', 'above average', 'we should watch a movie', 11000, 0),
        ('mom', 'you bad', 'take a walk', 12000, 1),
        ('mom', 'love you', 'find someone else', 1300, 0),
        ('mom', 'miss you', 'be better', 200, 0),
        ('mom', 'youre disapointment', 'you are articulate', 100, 1),
        ('sister', 'amazing job', 'ugly people bother me', 200, 1),
        ('sister', 'hate you', 'calm people are nice', 100, 0),
        ('sister', 'miss you', 'pretty people have nice faces', 50, 1),
        ('sister', 'wiz kid', 'you are pretty', 1000, 1),
        ('sister', 'love you', 'i like your face', 200, 0),
        ('father', 'amazing job', 'i walk and talk', 500, 0),
        ('father', 'proud you', 'you are amazing', 1000, 0),
        ('father', 'work harder', 'let us create a plan', 5000, 1),
        ('father', 'love you', 'plan to get together', 500, 0),
        ('father', 'dont quit','pretty people are more fun', 100, 0)]

# data_dev = [('professor', ['you', 'average'], ['pretty people are more fun'], 1000, 1), 
#             ('brother', ['dont', 'quit'], ['you are ugly'], 2000, 0),
#             ('mom', ['nice', 'haircut'], ['you are pretty'], 3000, 0),
#             ('sister', ['bad', 'clothes'], ['take a walk'], 4000, 1),
#             ('father', ['love', 'you'], ['you rock'], 5000, 0)]

df = pd.DataFrame(data, columns=['speaker', 'context_words', 'column_three', 'quant','sentiment'])
# df_dev = pd.DataFrame(data_dev, columns=['speaker', 'context_words', 'column_three', 'quant','sentiment']) 

CONTEXT_SIZE = 2 
EMBEDDING_DIM = 10


# context column - vocab to index - map to row
context_n, context_vocab = set(), set() 
for context in df['context_words']:
    words = context.split()
    context_n.add(context)
    for i in range(len(words)-CONTEXT_SIZE + 1):
        context_window = words[i: i + CONTEXT_SIZE]
        context_vocab.update(context_window) 
word_to_ix_context = {word: i for i, word in enumerate(context_vocab)}

def set_index(sentence, word_to_ix_context):
    words= sentence.split()
    indices = [word_to_ix_context[word] for word in words if word in word_to_ix_context]
    return indices
text_context_col = [set_index(row, word_to_ix_context) for row in df['context_words']]

# col3 column - vocab to index - map to row
col_n, col_three_vocab = set(), set() 
for col in df['column_three']:
    words = col.split()
    col_n.add(col)
    for i in range(len(words)-CONTEXT_SIZE + 1):
        context_window = words[i: i + CONTEXT_SIZE]
        col_three_vocab.update(context_window) 
word_to_ix_col_three = {word: i for i, word in enumerate(col_three_vocab)}

def set_index(sentence, word_to_ix_col_three):
    words= sentence.split()
    indices = [word_to_ix_col_three[word] for word in words if word in word_to_ix_col_three]
    return indices
text_col_three = [set_index(row, word_to_ix_col_three) for row in df['column_three']]

######################

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, word_to_ix_context, word_to_ix_col_three):
        self.df = df
        self.word_to_ix_context = word_to_ix_context
        self.word_to_ix_col_three = word_to_ix_col_three

    def __getitem__(self, index):
        context_indices = torch.tensor([self.word_to_ix_context[word] for word in self.df.iloc[index]['context_words']], dtype=torch.int64)
        speaker = torch.tensor(self.df.iloc[index]['speaker'], dtype=torch.long)
        col_three_indices = torch.tensor([self.word_to_ix_col_three[word] for word in self.df.iloc[index]['column_three']], dtype=torch.int64)
        quant = torch.tensor(self.df.iloc[index]['quant'], dtype=torch.float32)
        sentiment = torch.tensor(self.df.iloc[index]['sentiment'], dtype=torch.float32)
        return context_indices, speaker, col_three_indices, quant, sentiment
    
    def __len__(self):
        return len(self.df) 


#### 
# PAD 

    
##### 
# NN 

class NGramLanguageModeler(nn.Module):
    def __init__(self, num_speakers, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim + (context_size * embedding_dim) + 1, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        context_indices, speaker, col_three_indices, quant, sentiment = inputs
        context_embeds = self.word_embeddings(torch.tensor(context_indices)).view((1, -1))
        speaker_embed = self.speaker_embeddings(torch.tensor(speaker)).view((1, -1))
        col_three_embeds = self.word_embeddings(torch.tensor(col_three_indices)).view((1, -1))
        embeds_full = torch.cat((speaker_embed, context_embeds, col_three_embeds, torch.tensor([quant]).view((1, -1))), dim=1)
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs    


#### 
# test train split 

#### 
# Initialize Datasets and Dataloaders
train_dataset = SentimentDataset(df, word_to_ix_context, word_to_ix_col_three)
dev_dataset = SentimentDataset(df, word_to_ix_context, word_to_ix_col_three)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=4)

### 
# Train


###############################

# class SentimentDataset(torch.utils.data.Dataset):
#     def __init__(self, df, word_to_ix):
#         self.df = df
#         self.word_to_ix = word_to_ix

#     def __getitem__(self, index):
#         context_indices = [self.word_to_ix[word] for word in self.df.iloc[index]['context_words']]
#         speaker_code = self.df.iloc[index]['speaker_code']
#         sentiment = self.df.iloc[index]['sentiment']
#         return torch.tensor(context_indices), torch.tensor(speaker_code), torch.tensor(sentiment)

#     def __len__(self):
#         return len(self.df)
# class NGramLanguageModeler(nn.Module):
#     def __init__(self, num_speakers, vocab_size, embedding_dim, context_size):
#         super(NGramLanguageModeler, self).__init__()
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
#         self.linear1 = nn.Linear(embedding_dim + (context_size * embedding_dim), 128)
#         self.linear2 = nn.Linear(128, 1) 

#     def forward(self, inputs):
#         speaker_code, word_indices = inputs
#         word_embeds = self.word_embeddings(word_indices).view((1, -1))
#         speaker_embed = self.speaker_embeddings(speaker_code).view((1, -1))
#         embeds_full = torch.cat((speaker_embed, word_embeds), dim=1) 
#         out = F.relu(self.linear1(embeds_full))
#         out = self.linear2(out)
#         log_probs = torch.sigmoid(out) 
#         return log_probs

# Initialize Datasets and Dataloaders
# train_dataset = SentimentDataset(df, word_to_ix)
# dev_dataset = SentimentDataset(df_dev, word_to_ix)  

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=4)


# Initialize model, loss, optimizer
losses = []
val_losses = []
# Initialize model, loss, optimizer
model = NGramLanguageModeler(len(speaker_to_ix), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)



for epoch in range(100):
    total_loss = 0
    
    ##############
    num_correct_train = 0  # Track accuracy for training
    num_total_train = 0
    
    ###########
    
    for speaker, sentence, target in data:
        word_idxs = [word_to_ix[w] for w in sentence]
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        speaker_idxs = [speaker_to_ix[speaker]]
        speaker_idxs = torch.tensor(speaker_idxs, dtype=torch.long)
        model.zero_grad()
        
        log_probs = model((speaker_idxs, word_idxs))
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.float).resize_((1, 1)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
         ##############
        # Training Accuracy Calculation
        predicted = log_probs.round().item()  # Get the predicted class (0 or 1)
        num_correct_train += (predicted == target) 
        num_total_train += 1
        
        ###########
        

    losses.append(total_loss)
    model.eval()
    num_correct_val = 0  # Track accuracy for validation
    num_total_val = 0
    total_val_loss = 0  # We need to calculate this

    with torch.no_grad():
      for speaker, sentence, target in data_dev:
        word_idxs = [word_to_ix[w] for w in sentence]
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        speaker_idxs = [speaker_to_ix[speaker]]
        speaker_idxs = torch.tensor(speaker_idxs, dtype=torch.long)
        log_probs = model((speaker_idxs, word_idxs))
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.float).resize_((1, 1)))
   ########   ########   ########
    
        total_val_loss += loss.item() 
        
        #############################
        val_losses.append(loss.item())
        
       
      # Validation Accuracy Calculation
        predicted = log_probs.round().item()
        num_correct_val += (predicted == target)
        num_total_val += 1
        
    # Calculate and print accuracies
    train_accuracy = num_correct_train / num_total_train
    val_accuracy = num_correct_val / num_total_val

    print(f'Epoch {epoch + 1} | Train Loss: {total_loss:.4f} | Train Accuracy: {train_accuracy:.3f} | Val Loss: {total_val_loss:.4f} | Val Accuracy: {val_accuracy:.3f}') 
        #####
    
    
    
    
    print('train_loss == ', losses[-1], 'dev_loss == ', val_losses[-1])



for row in zip(losses, val_losses):
  print(row)