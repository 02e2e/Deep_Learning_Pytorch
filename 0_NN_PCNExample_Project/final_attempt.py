
#### attempt 1 on wedensday 

# Load Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import random
import string
import re
import pandas as pd
import numpy as np
import random

# Sentences (varying length 2-10 words)
words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "eats", "food"]
sentences = []
for _ in range(20):
    length = random.randint(2, 10)
    sentences.append(" ".join(random.sample(words, length)))

# Colors (8 different)
colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]

# Companies (varying length 1-4 words)
company_words = ["Apple California Proper", "Google", "Microsoft Seattle", "Amazon", "Tesla Texas", "SpaceX", "IBM", "Samsung Incoporated"]
companies = []
for _ in range(20):
    length = random.randint(1, 4)
    companies.append(" ".join(random.sample(company_words, length)))

# Quantity (discrete numeric)
quantity = np.random.randint(1, 101, 20)  # 1 to 100

# Target (binary 0/1)
target = np.random.randint(0, 2, 20)

# Create DataFrame
data = {
    "sentence": sentences,
    "color": np.random.choice(colors, 20),
    "company": companies,
    "quantity": quantity,
    "target": target
}
df = pd.DataFrame(data)

############# Data Above ########

#################################################################


df.dtypes
# identify numeric columns 
numeric_columns = df['quantity']
numeric_columns_values = numeric_columns.values
# identify categorical columns 
categorical_columns = ['color']
longer_cat_variables = ['sentence', 'company']
longer_cat_var_values = df.loc[:, ['sentence', 'company']].values




# preprocess data 
# Apply StandardScaler normalization to numeric columns
scaler = StandardScaler()
df['quantity'] = scaler.fit_transform(df['quantity'].values.reshape(-1, 1))



# label encoding our target variable to a 0/1 
le = LabelEncoder()
# encode = OneHotEncoder()
df['color'] = le.fit_transform(df['color'])
# df['procedureAnalysedMedia'] = encode.fit_transform(df['procedureAnalysedMedia'])

# target_column = df['procedureAnalysedMedia']

# Extract target variable values from the DataFrame
target_data = df['target'].values  # This creates a NumPy array of the target variable

# drop the target variable 
df.drop('target', axis=1, inplace=True) # these are one-hot encoded 
# target = 'target'
numeric_data = df['quantity'].values
multi_class_cat_data = df['color'].values



######################################################################

######## end of May 28th Tuesday ##########
EMBEDDING_DIM = 10 
from torch.nn.utils.rnn import pad_sequence

CONTEXT_SIZE = 2  
companies, company_vocab = [], []  
for company in df['company']:
    words = company.lower().split()
    companies.append(words)         # Store list of words for each company
    company_vocab.extend(words)     # Extend the vocabulary list with individual words
    for i in range(max(0, len(words) - CONTEXT_SIZE + 1)): 
        context_window = words[i: i + CONTEXT_SIZE]
        company_vocab.extend(context_window)   # Extend the vocabulary with words from the context window

word_to_ix = {word: i for i, word in enumerate(set(company_vocab))}  # Remove duplicates for indexing
MAX_LENGTH = 4  # Set a maximum length for company names
def sentence_to_indices(words, word_to_ix):
    indices = [word_to_ix[word] for word in words if word in word_to_ix]
    indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
    return indices

company_data = [sentence_to_indices(row, word_to_ix) for row in companies]

print(company_data)   # This should now give you the correct indices

company_data_padded = pad_sequence([torch.tensor(seq) for seq in company_data], batch_first=True, padding_value=0)
company_data_padded.shape
# torch.Size([20, 8])




####

CONTEXT_SIZE = 4
sentences, sentences_vocab = [], []  
for sentence in df['sentence']:
    words = sentence.lower().split()
    sentences.append(words)         # Store list of words for each company
    sentences_vocab.extend(words)     # Extend the vocabulary list with individual words
    for i in range(max(0, len(words) - CONTEXT_SIZE + 1)): 
        context_window = words[i: i + CONTEXT_SIZE]
        sentences_vocab.extend(context_window)   # Extend the vocabulary with words from the context window

word_to_ix1 = {word: i for i, word in enumerate(set(sentences_vocab))}  # Remove duplicates for indexing
MAX_LENGTH = 8 # Set a maximum length for company names
def sentence_to_indices(words, word_to_ix1):
    indices = [word_to_ix1[word] for word in words if word in word_to_ix1]
    indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
    return indices

sentence_data = [sentence_to_indices(row, word_to_ix1) for row in sentences]

print(sentence_data) 
sentence_data_padded = pad_sequence([torch.tensor(seq) for seq in sentence_data], batch_first=True, padding_value=0)

sentence_data_padded.shape
# torch.Size([20, 10])

# Concatenate along a new dimension (dim=1 to stack horizontally)
# combined_data = torch.cat((company_data_padded, sentence_data_padded), dim=1)

# print(combined_data.shape) 
##############
# New May 29th Tuesday 

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class WordEmbedDataset(Dataset):
    def __init__(self, sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target_data):
        self.sentence_data_padded = sentence_data_padded
        print("Sample of sentence_data_padded:", sentence_data_padded[0])
        
        self.company_data_padded = company_data_padded
        print("Sample of company_data_padded:", company_data_padded[0])
        
        self.numeric_data = numeric_data
        print("Sample of numeric_data:", numeric_data[0])
        
        self.multi_class_cat_data = multi_class_cat_data
        print("Sample of multi_class_cat_data:", multi_class_cat_data[0])
        
        self.target_data = target_data
        print("Sample of target_data:", target_data[0])
    
    def __getitem__(self, index):
        sentence_data_padded = torch.tensor(self.sentence_data_padded[index], dtype=torch.long)
        company_data_padded = torch.tensor(self.company_data_padded[index], dtype=torch.long)
        numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32)
        multi_class_cat_data = torch.tensor(self.multi_class_cat_data[index], dtype=torch.long)
        target = torch.tensor(self.target_data[index], dtype=torch.float32) # Adjust if necessary
        
        return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target
    
    def __len__(self):
        return len(self.sentence_data_padded)


# Create dataset
dataset = WordEmbedDataset(sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target_data)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Iterate through DataLoader to test
# for batch in dataloader:
#     sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = batch
#     print("Batch of sentence_data_padded:", sentence_data_padded)
#     print("Batch of company_data_padded:", company_data_padded)
#     print("Batch of numeric_data:", numeric_data)
#     print("Batch of multi_class_cat_data:", multi_class_cat_data)
#     print("Batch of target:", target)
#     break  # Print only the first batch to check

# Now we are here below 

def collate_fn(batch):
    # Unzip the batch
    sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = zip(*batch)
    
    # Stack numeric and categorical data
    sentence_data_padded = torch.stack(sentence_data_padded)
    company_data_padded = torch.stack(company_data_padded)
    numeric_data = torch.stack(numeric_data)
    multi_class_cat_data = torch.stack(multi_class_cat_data)
    target = torch.stack(target)
    
    return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target


# Iterate through DataLoader to test the collate_fn
for batch in dataloader:
    sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = batch
    
    # Print shapes
    print("Shape of sentence_data_padded:", sentence_data_padded.shape)
    print("Shape of company_data_padded:", company_data_padded.shape)
    print("Shape of numeric_data:", numeric_data.shape)
    print("Shape of multi_class_cat_data:", multi_class_cat_data.shape)
    print("Shape of target:", target.shape)
    
    # Print contents
    print("Batch of sentence_data_padded:", sentence_data_padded)
    print("Batch of company_data_padded:", company_data_padded)
    print("Batch of numeric_data:", numeric_data)
    print("Batch of multi_class_cat_data:", multi_class_cat_data)
    print("Batch of target:", target)
    
    break  # Print only the first batch to check














######## end of May 28th Tuesday ##########





# ######################################
# # Last week prior to May 28th 
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# class WordEmbedDataset(Dataset):
#     def __init__(self, sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data,target_data):
#         self.sentence_data_padded = sentence_data_padded
#         print("Sample of text_data_1:", sentence_data_padded[0])  # Print the first element

#         self.company_data_padded = company_data_padded
#         print("Sample of text_data_2:", company_data_padded[0])
        
#         self.numeric_data = numeric_data
#         print("Sample of numeric_features:", numeric_data[0])
        
#         self.multi_class_cat_data = multi_class_cat_data
#         print("Sample of numeric_features:", multi_class_cat_data[0])
        
#         self.target_data = target_data
#         print("Sample of target_data:", target_data[0])
      
      
#     def __getitem__(self, index):
#         sentence_data_padded = torch.tensor(self.sentence_data_padded[index], dtype=torch.long)
#         company_data_padded = torch.tensor(self.company_data_padded[index], dtype=torch.long)
#         multi_class_cat_data = torch.tensor(self.multi_class_cat_data[index], dtype=torch.int64) 
#         numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32)
#         target = torch.tensor(self.target_data[index], dtype=torch.float32)  # Adjust if necessary
#         return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target

    # ... (Your __len__ method remains the same)
    #     sentence_data_padded = torch.tensor(self.sentence_data_padded[index], dtype=torch.long)  # Specify long dtype
    #     company_data_padded = torch.tensor(self.company_data_padded[index], dtype=torch.long)  # Specify long dtype
    #     multi_class_cat_data = torch.tensor(self.multi_class_cat_data[index], dtype=torch.int64) 
    #     numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32) 
    #     target = self.target_data[index] # Add this line for the targets
    #     target = torch.tensor(target, dtype=torch.int64) # or .float32
    #     return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target 
    
    # def __len__(self):
    #     return len(self.sentence_data_padded)  # Assuming lists have the same length 

# dataset = WordEmbedDataset(sentence_data_padded, company_data_padded, df['quantity'].values, df['color'].values, df['target'].values)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 




#######################

####################################### May 17 Friday below 
# May 17 Friday
# updateing the dataset once again 

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# class WordEmbedDataset(Dataset):
#     def __init__(self, text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data):
#         self.text_data_1 = [torch.tensor(data, dtype=torch.long) for data in text_data_1]
#         self.text_data_2 = [torch.tensor(data, dtype=torch.long) for data in text_data_2]
#         self.numeric_data = [torch.tensor(data, dtype=torch.float32) for data in numeric_data]
#         self.multi_class_cat_data = [torch.tensor(data, dtype=torch.int64) for data in multi_class_cat_data]
#         self.target_data = [torch.tensor(data, dtype=torch.int64) for data in target_data]
        
#         print("Sample of text_data_1:", self.text_data_1[0])  # Print the first element
#         print("Sample of text_data_2:", self.text_data_2[0])
#         print("Sample of numeric_features:", self.numeric_data[0])
#         print("Sample of multi_class_cat_data:", self.multi_class_cat_data[0])
#         # print("Sample of target_data:", self.target_data[0])

#     def __getitem__(self, index):
#         # Retrieve data for the given index
#         text_data_1 = self.text_data_1[index]
#         text_data_2 = self.text_data_2[index]
#         numeric_data = self.numeric_data[index]
#         multi_class_cat_data = self.multi_class_cat_data[index]
#         target = self.target_data[index]
        
#         # Ensure that text_data_1 and text_data_2 are lists of sequences
#         if not isinstance(text_data_1, list):
#             text_data_1 = [text_data_1]
#         if not isinstance(text_data_2, list):
#             text_data_2 = [text_data_2]
        
#         print("Type of text_data_1:", type(text_data_1))
#         print("Type of text_data_2:", type(text_data_2))
#         print("Length of text_data_1:", len(text_data_1))
#         print("Length of text_data_2:", len(text_data_2))
        
#         # Pad text data sequences
#         text_data_1_padded = pad_sequence([torch.tensor(seq) for seq in text_data_1], batch_first=True, padding_value=0)
#         text_data_2_padded = pad_sequence([torch.tensor(seq) for seq in text_data_2], batch_first=True, padding_value=0)
        
#         return text_data_1_padded, text_data_2_padded, numeric_data, multi_class_cat_data, target

# def collate_fn(batch):
#     # Unzip the batch
#     text_data_1_padded, text_data_2_padded, numeric_data, multi_class_cat_data, target = zip(*batch)
    
#     # Stack numeric and categorical data
#     numeric_data = torch.stack(numeric_data)
#     multi_class_cat_data = torch.stack(multi_class_cat_data)
#     target = torch.stack(target)
    
#     return text_data_1_padded, text_data_2_padded, numeric_data, multi_class_cat_data, target



# dataset = WordEmbedDataset(text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data)
# dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

# # Iterate over batches
# for i, batch in enumerate(dataloader):
#     text_data_1_padded, text_data_2_padded, numeric_data, multi_class_cat_data, target = batch
#     print(f"Batch {i+1}:")
#     print(f"  Text Data 1 shape: {text_data_1_padded[0].shape}")  # Accessing shape of the first tensor
#     print(f"  Text Data 2 shape: {text_data_2_padded[0].shape}")  # Accessing shape of the first tensor
#     print(f"  Numeric Data shape: {numeric_data.shape}")
#     print(f"  Multi-class Cat Data shape: {multi_class_cat_data.shape}")
#     print(f"  Target shape: {target.shape}")







################ FRIDAY ABOVE MAY 17 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Example NGram Language Modeler
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = torch.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

# Assuming vocab_size and embedding_dim are defined
vocab_size = 100  # Example vocab size
embedding_dim = 10  # Example embedding dimension
context_size = 3  # Example context size for NGram model

# Instantiate model, loss function, and optimizer
model = NGramLanguageModeler(vocab_size, embedding_dim, context_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example data (using your dataset and dataloader)
text_data_1 = [[1, 2, 3], [4, 5, 6]]
text_data_2 = [[7, 8, 9], [10, 11, 12]]
numeric_data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
multi_class_cat_data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
target_data = [0, 1]

dataset = WordEmbedDataset(text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

# Training loop (example)
for epoch in range(10):  # Example number of epochs
    total_loss = 0
    for text_data_1, text_data_2, numeric_data, multi_class_cat_data, target in dataloader:
        # Assuming text_data_1 is used for the NGram modeler input
        model.zero_grad()
        log_probs = model(text_data_1)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Note: Adjust the training loop and model according to your specific task and requirements.

######################################

# from torch.nn.utils.rnn import pad_sequence

# def pad_collate(batch):
#     # Sample batch: [([text_indices], [other_text_indices], numeric_features, label), ...]
#     text_1, text_2, numeric_data, multi_class_cat_data, labels = zip(*batch)  # Unpack the batch assuming the data point is a tuple 

#     padded_text_1 = pad_sequence(text_1, batch_first=True, padding_value=OOV_INDEX)
#     padded_text_2 = pad_sequence(text_2, batch_first=True, padding_value=OOV_INDEX)

#     # Convert to PyTorch tensors
#     numeric_data = torch.stack(numeric_data) 
#     multi_class_cat_data = torch.stack(multi_class_cat_data) 
#     labels = torch.tensor(labels) # convert to pytorch tensors 

#     return padded_text_1, padded_text_2, numeric_data, multi_class_cat_data, labels

# ##########################
# # create an object representing the instance of the custom dataset class
# my_dataset = WordEmbedDataset(text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data) 
# # created a dataloader object ready to iterate over your dataset batches. 
# dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate) 

































#################################
import torch.nn as nn

class SimpleMultiModalNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_features, cat_features, output_dim):
        super().__init__()

        # Embedding layers for text features (adjust dimensions if needed)
        self.embedding_1 = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_2 = nn.Embedding(vocab_size, embedding_dim)

        # Linear layers for numerical features
        self.linear_numeric = nn.Linear(num_features, 32) 

        # Linear layers for categorical features (one-hot encoding assumed)
        self.linear_cat = nn.Linear(cat_features, 16)  

        # Hidden layers (adjust these as needed)
        self.hidden_1 = nn.Linear(32 + 32 + 16, 64)  # Concatenated embeddings and features
        self.relu = nn.ReLU()
        self.hidden_2 = nn.Linear(64, 32)

        # Output layer (since you have a binary classification problem)
        self.output = nn.Linear(32, output_dim)  

    def forward(self, text_data_1, text_data_2, numeric_data, cat_data):
        embedded_1 = self.embedding_1(text_data_1)
        embedded_2 = self.embedding_2(text_data_2)

        # Optionally average or concatenate word embeddings here

        numeric_out = self.relu(self.linear_numeric(numeric_data))
        cat_out = self.relu(self.linear_cat(cat_data))

        combined = torch.cat([embedded_1, embedded_2, numeric_out, cat_out], dim=1)
        hidden_out = self.relu(self.hidden_1(combined))
        hidden_out = self.relu(self.hidden_2(hidden_out))
        output = self.output(hidden_out)
        return output


# ... (Your imports, data loading, and preprocessing)

# NO LONGER SPLITTING HERE!
import torch.optim as optim
num_epochs = 10
learning_rate = 0.001
batch_size = 32  # Added batch size
# Initialize Model, Loss Function, and Optimizer
vocab_size = len(vocab) 
num_features = numeric_data.shape[1] 
cat_features = multi_class_cat_data.shape[1] 

model = SimpleMultiModalNN(vocab_size, EMBEDDING_DIM, num_features, cat_features, output_dim=1)  
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create single dataloader object 
dataloader = DataLoader(
    WordEmbedDataset(text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data), 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=pad_collate) 

# Modified Training Loop (No separate validation)
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  
        for text_data_1, text_data_2, numeric_data, multi_class_cat_data, targets in dataloader: 
            optimizer.zero_grad()  
            outputs = model(text_data_1, text_data_2, numeric_data, multi_class_cat_data)  
            loss = criterion(outputs.squeeze(1), targets.float())  
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


train_model(model, dataloader, criterion, optimizer, num_epochs) 



##########################################
# # train test split version
# # Hyperparameters
# num_epochs = 10
# learning_rate = 0.001
# batch_size = 32  # Added batch size

# # Initialize Model, Loss Function, and Optimizer
# vocab_size = len(vocab)  # Ensure this is accurate
# num_features = X_train.shape[1]  # Number of numerical features
# cat_features = multi_class_cat_data.shape[1]  # Number of categorical features

# model = SimpleMultiModalNN(vocab_size, embedding_dim, num_features, cat_features, output_dim=1)  
# criterion = nn.BCEWithLogitsLoss() 
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Create dataloader objects 
# train_dataloader = DataLoader(
#     WordEmbedDataset(text_data_1, text_data_2, X_train, multi_class_cat_data, y_train), 
#     batch_size=batch_size, 
#     shuffle=True, 
#     collate_fn=pad_collate) 

# test_dataloader = DataLoader(
#     WordEmbedDataset(text_data_1, text_data_2, X_test, multi_class_cat_data, y_test), 
#     batch_size=batch_size, 
#     shuffle=False, 
#     collate_fn=pad_collate)  

# # Training and Evaluation Loop
# def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode
#         for text_1, text_2, numeric_data, multi_class_cat_data, targets in train_loader: 
#             optimizer.zero_grad()  
#             outputs = model(text_1, text_2, numeric_data, multi_class_cat_data)  
#             loss = criterion(outputs.squeeze(1), targets.float())  
#             loss.backward()
#             optimizer.step()

#         model.eval()  # Set to evaluation mode
#         with torch.no_grad():
#           correct = 0
#           total = 0
#           for text_1, text_2, numeric_data, multi_class_cat_data, targets in test_loader:
#             outputs = model(text_1, text_2, numeric_data, multi_class_cat_data)
#             _, predicted = torch.max(outputs.data, dim=1)  # Get predicted class
#             total += targets.size(0)  
#             correct += (predicted == targets).sum().item()  
        
#         accuracy = 100 * correct / total
#         print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%')

# train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs) 

#################################################




############################

# neural network 


# attempt 2
import torch.nn as nn
import torch.nn.functional as F  

class WordEmbeddingModel(nn.Module):
    def __init__(self, num_numeric_features):
        super().__init__()
        EMBEDDING_DIM = 10
        CONTEXT_SIZE_1 = 1 
        CONTEXT_SIZE_2 = 1

        self.linear1 = nn.Linear(2 * EMBEDDING_DIM + num_numeric_features, 128)  
        self.linear2 = nn.Linear(128, 1)  

    def forward(self, text_1, text_2, numeric_features):
        embed_1 = self.contextual_embedding(text_1, CONTEXT_SIZE_1)
        embed_2 = self.contextual_embedding(text_2, CONTEXT_SIZE_2)
        combined_embeds = torch.cat((embed_1, embed_2, numeric_features), dim=1) 
        out = F.relu(self.linear1(combined_embeds)) 
        out = self.linear2(out)
        log_probs = torch.sigmoid(out) 
        return log_probs 
    
    def contextual_embedding(self, text_data, context_size):
        embeds = []
        for i in range(len(text_data)):
            context_window = text_data[max(0, i - context_size) : i + context_size + 1]
            embed = F.embedding(torch.tensor(context_window), nn.Embedding(max(context_window) + 1, self.embedding_dim))
            embeds.append(embed.mean(dim=0))  # Average over context window
        return torch.stack(embeds)  

model = WordEmbeddingModel(len(df.columns) - (len(text_data_1[0]) + 1)) 

#######

# losses = []
# val_losses = []
# loss_function = nn.BCELoss()
# model = WordEmbeddingModel(len(df.columns) - (len(text_data_1[0]) + 1))  EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)


import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score  

# ... (Your data preparation, dataset, and model definition remain unchanged) ...

# Assuming your data is prepared and accessible within the dataloader
loss_function = nn.BCEWithLogitsLoss()  # Assuming binary classification
model = WordEmbeddingModel(len(df.columns) - (len(text_data_1[0]) + 1)) 
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust optimizer if needed

losses = []
for epoch in range(100):
    total_loss = 0
    for text_1, text_2, numeric_features, target in dataloader:
        # Data Processing (Ensure this aligns with output from your dataloader)
        text_1 = text_1.long()    # Assuming word indices
        text_2 = text_2.long()    # Assuming word indices
        numeric_features = numeric_features.float()  # Assuming numeric features
        target = target.float().unsqueeze(1)  

        # Training Step
        model.zero_grad()
        log_probs = model(text_1, text_2, numeric_features)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
        
print('Training Completed!')  



######











