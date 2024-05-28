
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

# file path
file_path = "CleanDF.csv" 

df = pd.read_csv(file_path)

# Set the maximum number of columns to display to None
pd.set_option('display.max_columns', None)
df.head()
df = df.drop(columns='Unnamed: 0')

df.dtypes
# identify numeric columns 
numeric_columns = df.select_dtypes('number').columns
# identify categorical columns 
categorical_columns = df.select_dtypes('object').columns
# look at the cardinality of the categorical variables
df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')


########### 3/27 -- 2:19PM #######
# add two columns to the data to try the word embedding 

# Function to generate random words
def generate_random_words(num_words):
    words = []
    for _ in range(num_words):
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 5)))
        words.append(word)
    return ' '.join(words)

# Generate new DataFrame
num_rows = len(df)  # Assuming you want the new DataFrame to have the same length as the existing one
titles = [generate_random_words(random.randint(1, 2)) for _ in range(num_rows)]
names = [generate_random_words(random.randint(1, 3)) for _ in range(num_rows)]

new_df = pd.DataFrame({'title': titles, 'name': names})

# add values to dataframe to act as mock variables to 
# mult-word categorical variables added 

df['observedPropertyDeterminandCode'] = new_df['title']# title 4 - 5 words random long 
df['resultUom'] = new_df['name']# name 2-3 words random long 
# Display the updated DataFrame
print(df)

# preprocess data 
# Apply StandardScaler normalization to numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# label encoding our target variable to a 0/1 
le = LabelEncoder()
# encode = OneHotEncoder()
df['procedureAnalysedMedia'] = le.fit_transform(df['procedureAnalysedMedia'])
# df['procedureAnalysedMedia'] = encode.fit_transform(df['procedureAnalysedMedia'])

# target_column = df['procedureAnalysedMedia']

# Extract target variable values from the DataFrame
target_data = df['procedureAnalysedMedia'].values  # This creates a NumPy array of the target variable

# drop the target variable 
df.drop('procedureAnalysedMedia', axis=1, inplace=True) # these are one-hot encoded 
target = 'procedureAnalysedMedia'

# label encode regular categorical features 
df['parameterWaterBodyCategory'] = le.fit_transform(df['parameterWaterBodyCategory'])

df['procedureAnalysedFraction'] = le.fit_transform(df['procedureAnalysedFraction'])

df['parameterSamplingPeriod'] = le.fit_transform(df['parameterSamplingPeriod'])

df['waterBodyIdentifier'] = le.fit_transform(df['waterBodyIdentifier'])

df['Country'] = le.fit_transform(df['Country'])


# identify numeric columns 
numeric_features = df.select_dtypes('number').columns
###### get the numeric column data 
numeric_data = df.loc[:, numeric_features].values
print(type(numeric_data))
# <class 'numpy.ndarray'>
### get the categorical column data 
# identify categorical columns 
multi_class_features = ['waterBodyIdentifier','Country','parameterSamplingPeriod','parameterWaterBodyCategory']

multi_class_cat_data = df.loc[:, multi_class_features].values
print(type(multi_class_cat_data))



# look at the cardinality of the categorical variables
df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')

# getting the names of features into lists for later use 
# categorical_columns_list = list(categorical_features)


numeric_columns_list = list(numeric_features)
longer_cat_col_list = list(df[['observedPropertyDeterminandCode','resultUom']])


CONTEXT_SIZE = 1
EMBEDDING_DIM = 10 

def extract_context_words(text_series, context_size):
    vocab = set()
    for text in text_series:
        words = text.split()  
        for i in range(len(words) - context_size + 1):
            context_window = words[i : i + CONTEXT_SIZE]
            vocab.update(context_window)  # Update with individual words
    return vocab

vocab = extract_context_words(df['observedPropertyDeterminandCode'], CONTEXT_SIZE)

vocab2 = extract_context_words(df['resultUom'], CONTEXT_SIZE) 

word_to_ix = {word: i for i, word in enumerate(vocab)}
word_to_ix2= {word: i for i, word in enumerate(vocab2)}


OOV_TOKEN = "<UNK>"
OOV_INDEX = 0  # Assign index 0 to the OOV token 
vocab.add(OOV_TOKEN)  # Add to the vocabulary
word_to_ix = {word: i for i, word in enumerate(vocab)} 


def sentence_to_indices(sentence, word_to_ix):
    words = sentence.split() 
    indices = [word_to_ix[word] for word in words if word in word_to_ix]
    return indices

# Preprocess Text Columns
text_data_1 = [sentence_to_indices(row, word_to_ix) for row in df['observedPropertyDeterminandCode']]
text_data_2 = [sentence_to_indices(row, word_to_ix2) for row in df['resultUom']]


######
# check 
print("Contents of an inner list:", text_data_1[0][:5])  # Print the first 5 elements should be integers
print("Contents of an inner list:", text_data_2[0][:5])  # Print the first 5 elements
print("Sample of target_data:", target_data [0])
print(type(target_data)) # numpy array with a 1 or 0 
print(type(target_data))  # should be a list 

###############
import numpy as np 
from sklearn.model_selection import train_test_split

# ... (rest of your imports and data loading/preprocessing from previous responses) ...

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     numeric_data, multi_class_cat_data, target_data, test_size=0.2, random_state=42
# ) 
##################


class WordEmbedDataset(Dataset):
    def __init__(self, text_data_1, text_data_2, numeric_data, multi_class_cat_data,target_data):
        self.text_data_1 = text_data_1
        print("Sample of text_data_1:", text_data_1[0])  # Print the first element

        self.text_data_2 = text_data_2
        print("Sample of text_data_2:", text_data_2[0])
        self.numeric_data = numeric_data
        print("Sample of numeric_features:", numeric_data[0])
        self.multi_class_cat_data = multi_class_cat_data
        print("Sample of numeric_features:", multi_class_cat_data[0])
        self.target_data = target_data
        # print("Sample of target_data:", target_data[0])
      
      
    def __getitem__(self, index):
        text_data_1 = torch.tensor(self.text_data_1[index], dtype=torch.long)  # Specify long dtype
        text_data_2 = torch.tensor(self.text_data_2[index], dtype=torch.long)  # Specify long dtype
        multi_class_cat_data = torch.tensor(self.multi_class_cat_data[index], dtype=torch.int64) 
        numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32) 
        # target = self.target_data[index]
        target = torch.tensor(self.target_data[index], dtype=torch.int64)
        return text_data_1, text_data_2, numeric_data, multi_class_cat_data, target 
    
    def __len__(self):
        return len(self.text_data_1)  # Assuming lists have the same length 
  

######################################

from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    # Sample batch: [([text_indices], [other_text_indices], numeric_features, label), ...]
    text_1, text_2, numeric_data, multi_class_cat_data, labels = zip(*batch)  # Unpack the batch assuming the data point is a tuple 

    padded_text_1 = pad_sequence(text_1, batch_first=True, padding_value=OOV_INDEX)
    padded_text_2 = pad_sequence(text_2, batch_first=True, padding_value=OOV_INDEX)

    # Convert to PyTorch tensors
    numeric_data = torch.stack(numeric_data) 
    multi_class_cat_data = torch.stack(multi_class_cat_data) 
    labels = torch.tensor(labels) # convert to pytorch tensors 

    return padded_text_1, padded_text_2, numeric_data, multi_class_cat_data, labels

##########################
# create an object representing the instance of the custom dataset class
my_dataset = WordEmbedDataset(text_data_1, text_data_2, numeric_data, multi_class_cat_data, target_data) 
# created a dataloader object ready to iterate over your dataset batches. 
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate) 

































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











