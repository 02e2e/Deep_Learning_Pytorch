



import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

data = [('professor', ['you', 'smell'], 1), 
        ('professor', ['you', 'fail'], 1), 
        ('professor', ['youre', 'bad'], 1),
        ('professor', ['above', 'average'], 0),
        ('professor', ['hate', 'you'], 1),
        ('professor', ['wiz', 'kid'], 0),
        ('professor', ['amazing', 'job'], 0),
        ('brother', ['great', 'job'], 1),
        ('brother', ['wiz', 'kid'], 1),
        ('brother', ['you', 'fail'], 0),
        ('brother', ['hate', 'you'], 0),
        ('brother', ['you', 'smell'], 0),
        ('mom', ['you', 'smell'], 0),
        ('mom', ['above', 'average'], 0),
        ('mom', ['you', 'bad'], 1),
        ('mom', ['love', 'you'], 0),
        ('mom', ['miss', 'you'], 0),
        ('mom', ['youre', 'disapointment'], 1),
        ('sister', ['amazing', 'job'], 1),
        ('sister', ['hate', 'you'], 0),
        ('sister', ['miss', 'you'], 1),
        ('sister', ['wiz', 'kid'], 1),
        ('sister', ['love', 'you'], 0),
        ('father', ['amazing', 'job'], 0),
        ('father', ['proud', 'you'], 0),
        ('father', ['work', 'harder'], 1),
        ('father', ['love', 'you'], 0),
        ('father', ['dont', 'quit'], 0)]

data_dev = [('professor', ['you', 'average'], 1), 
            ('brother', ['dont', 'quit'], 0),
            ('mom', ['nice', 'haircut'], 0),
            ('sister', ['bad', 'clothes'], 1),
            ('father', ['love', 'you'], 0)]
             
df = pd.DataFrame(data, columns=['speaker', 'context_words', 'sentiment'])
df_dev = pd.DataFrame(data_dev, columns=['speaker', 'context_words', 'sentiment']) 

CONTEXT_SIZE = 2 
EMBEDDING_DIM = 10

# Create vocabulary (from both datasets)
vocab = set()
for df in [df, df_dev]:  
    for row in df.itertuples(index=False):
        for word in row.context_words:
            vocab.add(word)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# Encode speakers
speaker_to_ix = {speaker: i for i, speaker in enumerate(df['speaker'].unique())}  # Efficient using unique speakers
df['speaker_code'] = df['speaker'].map(speaker_to_ix)
df_dev['speaker_code'] = df_dev['speaker'].map(speaker_to_ix)




class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, word_to_ix):
        self.df = df
        self.word_to_ix = word_to_ix

    def __getitem__(self, index):
        context_indices = [self.word_to_ix[word] for word in self.df.iloc[index]['context_words']]
        speaker_code = self.df.iloc[index]['speaker_code']
        sentiment = self.df.iloc[index]['sentiment']
        return torch.tensor(context_indices), torch.tensor(speaker_code), torch.tensor(sentiment)

    def __len__(self):
        return len(self.df)
class NGramLanguageModeler(nn.Module):
    def __init__(self, num_speakers, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim + (context_size * embedding_dim), 128)
        self.linear2 = nn.Linear(128, 1) 

    def forward(self, inputs):
        speaker_code, word_indices = inputs
        word_embeds = self.word_embeddings(word_indices).view((1, -1))
        speaker_embed = self.speaker_embeddings(speaker_code).view((1, -1))
        embeds_full = torch.cat((speaker_embed, word_embeds), dim=1) 
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out) 
        return log_probs

# Initialize Datasets and Dataloaders
train_dataset = SentimentDataset(df, word_to_ix)
dev_dataset = SentimentDataset(df_dev, word_to_ix)  

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=4)


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