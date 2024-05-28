
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



# identify numeric columns 
numeric_features = df.select_dtypes('number').columns
# identify categorical columns 
categorical_features= df.select_dtypes('object').columns
# look at the cardinality of the categorical variables
df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')

# getting the names of features into lists for later use 
categorical_columns_list = list(categorical_features)


numeric_columns_list = list(numeric_features)
longer_cat_col_list = list(df[['observedPropertyDeterminandCode','resultUom']])



# # Monday April 29th , April 30th 
# CONTEXT_SIZE = 2
# EMBEDDING_DIM = 10 
# # vocabularies for each  longer_cat variables
# observes, observe_vocabs = set(), set()
# for row in df['observedPropertyDeterminandCode']:
#     words = row.split()
#     observes.add(row)
#     for i in range(len(words) - CONTEXT_SIZE + 1): 
#         context_window = words[i : i + CONTEXT_SIZE]
#         observe_vocabs.update(context_window)
# word_to_ix = {word: i for i, word in enumerate(observe_vocabs)}
# ix_to_word= {i: word for word, i in enumerate(observes)}


# CONTEXT_SIZE = 3
# EMBEDDING_DIM = 10 
# results, results_vocab = set(), set() 
# # results stores the unproceessed sentences
# # results_vocab build a vocabulary of words by extracting context windows in context-size =3 from each sentence 
# for row in df['resultUom']:
#     words = row.split()
#     results.add(row) # store the full row 
#     for i in range(len(words) - CONTEXT_SIZE + 1): 
#         context_window = words[i : i + CONTEXT_SIZE] # add words to the context window
#         results_vocab.update(context_window)
# word_to_ix2= {word: i for i, word in enumerate(results_vocab)} # maps each unique word in the vocabulary to a numerical index, this is what will be used for word embedding in the neural network 
# ix_to_word2= {i: word for word, i in enumerate(results)} # maps indices back to the original sentences this has limited use for word embedding

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


class WordEmbedDataset(Dataset):
    def __init__(self, text_data_1, text_data_2, numeric_features, target_data, df):
        self.text_data_1 = text_data_1
        self.text_data_2 = text_data_2
        self.numeric_features = numeric_features 
        self.target_data = target_data

    def __getitem__(self, index):
        text_data_1 = self.text_data_1[index]
        text_data_2 = self.text_data_2[index]
        numeric_data = self.df.loc[index, self.numeric_features].values  # Fetch from DataFrame
        target = self.target_data[index]
        return text_data_1, text_data_2, numeric_data, target 

    def __len__(self):
        return len(self.text_data_1)  # Assuming lists have the same length 

######################################

from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    # Sample batch: [([text_indices], [other_text_indices], numeric_features, label), ...]
    text_1, text_2, numeric_features, labels = zip(*batch)  # Unpack the batch assuming the data point is a tuple 

    padded_text_1 = pad_sequence(text_1, batch_first=True, padding_value=OOV_INDEX)
    padded_text_2 = pad_sequence(text_2, batch_first=True, padding_value=OOV_INDEX)

    # Convert to PyTorch tensors
    numeric_features = torch.stack(numeric_features) 
    labels = torch.tensor(labels) # convert to pytorch tensors 

    return padded_text_1, padded_text_2, numeric_features, labels

##########################
# create an object representing the instance of the custom dataset class
my_dataset = WordEmbedDataset(text_data_1, text_data_2, numeric_features, target_data, df) 
# created a dataloader object ready to iterate over your dataset batches. 
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate) 

############################

# neural network 


import torch.nn as nn
import torch.nn.functional as F  


# class WordEmbeddingModel(nn.Module):
#     def __init__(self, embedding_dim, num_numeric_features, context_sizes):
#         super().__init__()
#         self.embedding_dim = embedding_dim 
#         self.context_sizes = context_sizes

#         # Embedddings for numerical features (if needed) ...

#         self.linear1 = nn.Linear(2 * embedding_dim + num_numeric_features, 128)  
#         self.linear2 = nn.Linear(128, 1)  

#     def forward(self, text_1, text_2, numeric_features):
#         # Context-aware embeddings 
#         embed_1 = self.contextual_embedding(text_1, self.context_sizes[0])
#         embed_2 = self.contextual_embedding(text_2, self.context_sizes[1])

#         # Assuming embeddings need to be flattened (adjust if needed)
#         embed_1 = embed_1.view((1, -1)) 
#         embed_2 = embed_2.view((1, -1))  

#         combined_embeds = torch.cat((embed_1, embed_2, numeric_features), dim=1) 
#         out = F.relu(self.linear1(combined_embeds)) 
#         out = self.linear2(out)
#         log_probs = torch.sigmoid(out) 
#         return log_probs 
    
#     def contextual_embedding(self, text_data, context_size):
#         embeds = []
#         for i in range(len(text_data)):
#             context_window = text_data[max(0, i - context_size) : i + context_size + 1]
#             embed = F.embedding(torch.tensor(context_window), nn.Embedding(max(context_window) + 1, self.embedding_dim))
#             embeds.append(embed.mean(dim=0))  # Average over context window
#         return torch.stack(embeds) 


# attempt 2
import torch.nn as nn
import torch.nn.functional as F  

class WordEmbeddingModel(nn.Module):
    def __init__(self, num_numeric_features):
        super().__init__()
        EMBEDDING_DIM = 10
        CONTEXT_SIZE_1 = 1 
        CONTEXT_SIZE_2 = 3 

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

######


import torch.nn as nn
import torch.optim as optim 
from sklearn.model_selection import KFold 

#  ... Your 'WordEmbeddingModel' definition ...

def train_epoch(model, dataloader, optimizer, loss_function):
    model.train()  
    total_loss = 0

    for text_1, text_2, numeric_features, target in dataloader:
        optimizer.zero_grad()
        output = model(text_1, text_2, numeric_features)
        loss = loss_function(output, target.float().unsqueeze(1)) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  
    return total_loss / len(dataloader)  

def evaluate(model, dataloader, loss_function):
    model.eval() 
    total_loss = 0
    with torch.no_grad():
        for text_1, text_2, numeric_features, target in dataloader:
            output = model(text_1, text_2, numeric_features)
            loss = loss_function(output, target.float().unsqueeze(1))  
            total_loss += loss.item()
    return total_loss / len(dataloader)  

# Cross-validation
kf = KFold(n_splits=5)  # Example: 5-fold cross-validation
loss_function = nn.BCEWithLogitsLoss()  

for fold, (train_idx, val_idx) in enumerate(kf.split(X)): # Assuming 'X' contains your relevant data
    model = WordEmbeddingModel(len(numeric_features))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create train and validation dataloaders using the indices

    for epoch in range(10): 
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_function)
        val_loss = evaluate(model, val_dataloader, loss_function)
        print(f'Fold {fold + 1}, Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}') 























# ## numerical to tensor 
# import torch.nn.utils as rnn 
# numeric_tensor = torch.tensor(df[numeric_features].values, dtype=torch.float)
# numeric_tensor.shape
# # torch.Size([19427, 22])

# here up is fine good to go 
######################


# class NGramLanguageModeler(nn.Module):
#     def __init__(self, speaker_size, vocab_size, embedding_dim, context_size):
#         super(NGramLanguageModeler, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.embeddings2 = nn.Embedding(speaker_size, embedding_dim)
#         self.linear1 = nn.Linear(embedding_dim+(context_size * embedding_dim), 128)
#         self.linear2 = nn.Linear(128, 1)
#     def forward(self, inputs):
#         speaker,sentence = inputs
#         sentence_embed = self.embeddings(sentence).view((1, -1))
#         speaker_embed = self.embeddings2(speaker).view((1, -1))
#         embeds_full = torch.cat((speaker_embed,sentence_embed), -1) 
#         out = F.relu(self.linear1(embeds_full))
#         out = self.linear2(out)
#         log_probs = torch.sigmoid(out)
#         return log_probs















# ######################



# # ... (Define pad_collate function) ...

# # Dataset and Dataloader
# dataset = TensorDataset(text_data_1, text_data_2, numeric_features_tensor, labels_tensor) 
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)












# # class WordEmbeddingModel(nn.Module):
# #     def __init__(self, vocab_size, EMBEDDING_DIM, num_numeric_features, hidden_dim):
# #         super().__init__()
# #         self.embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
# #         self.linear1 = nn.Linear(EMBEDDING_DIM + num_numeric_features, hidden_dim)  
# #         self.linear2 = nn.Linear(hidden_dim, 1)  # Output layer for classification

# #     def forward(self, text, numeric_features):
# #         embeds = self.embeddings(text).mean(dim=1)  # Average word embeddings in a sentence
# #         combined_features = torch.cat((embeds, numeric_features), dim=1)
# #         out = F.relu(self.linear1(combined_features))
# #         out = self.linear2(out)
# #         return torch.sigmoid(out) 
















# from torch.nn.utils.rnn import pad_sequence

# MAX_SEQUENCE_LENGTH = 10  # Adjust this based on your data
# PAD_VALUE = 0

# def pad_collate(batch):
#     # 'batch' is a list of examples (e.g., [(text_indices, label), ...])
#     text_indices_list = [item[0] for item in batch]
#     labels = [item[1] for item in batch]

#     padded_text = pad_sequence(text_indices_list, batch_first=True, padding_value=PAD_VALUE)
#     return padded_text, torch.tensor(labels) 

# # Preprocess for Text Data and Create Tensors
# text_data_1 = []
# text_data_2 = []
# labels = target_data
# # Assuming this is your target variable

# for row in df['observedPropertyDeterminandCode']:
#     words = row.split()  # Adjust tokenization if needed
#     indices = [word_to_ix[word] for word in words if word in word_to_ix]
#     text_data_1.append(indices)

# for row in df['resultUom']:
#     words = row.split()  # Adjust tokenization if needed
#     indices = [word_to_ix2[word] for word in words if word in word_to_ix2]
#     text_data_2.append(indices)

# text_data_1_tensor = torch.tensor(text_data_1)
# text_data_2_tensor = torch.tensor(text_data_2)
# labels_tensor = torch.tensor(labels)

# # Combine features into a single tensor
# all_features = torch.cat((numeric_features_tensor, text_data_1_tensor, text_data_2_tensor), dim=1)

# # Create DataLoader
# dataset = TensorDataset(all_features, labels_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)









# ###################################################################################################
# # Step 2 4/23 - covert variables to tensors
# import torch.nn.utils.rnn as rnn
# # convert the numeric columns to a tensor - currently a pandas dataframe 
# numeric_tensor = torch.tensor(df[numeric_columns_list].values, dtype=torch.float) 
# # check shape of numeric_tensor
# numeric_tensor.shape 
# # torch.Size([19427, 22])
# ###########
# # Convert categorical columns to a tensor
# categorical_tensor = [
#     rnn.pad_sequence([
#         torch.tensor([word_to_ix[col].get(word, 0) for word in re.sub(r'[^\w\s]', '', row).lower().split()])
#         for row in df[col]
#     ]) for col in longer_cat_col_list
# ]
# # check shape of categorical tensor
# for i, tensor in enumerate(categorical_tensor):
#     print(f"Tensor {i} shape: {tensor.size()}") 
# # Tensor 0 shape: torch.Size([2, 19427])
# # Tensor 1 shape: torch.Size([3, 19427])
# ###########


# # ###########were having an error here ###########

# ###########
# #Convert one-hot encoded columns to a tensor
# import torch.nn.functional as F 
# one_hot_encoded_columns_list = {}
# for col in categorical_columns_list:
#     if col not in longer_cat_col_list:
#         unique_values = df[col].unique().tolist()  # Convert to list
#         # unique_values = df[col].unique()
#         # one_hot_encoded_columns_list[col] = F.one_hot(torch.tensor([df[col].values]), len(unique_values))
#         one_hot_encoded_columns_list[col] = F.one_hot(torch.tensor([unique_values.index(value) for value in df[col]]), len(unique_values))
# # Convert the dictionary view object to a list of tensors
# one_hot_encoded_tensors = list(one_hot_encoded_columns_list.values())
# # check shape of one-hot encoded tensors 
# for i, tensor in enumerate(one_hot_encoded_tensors):
#     print(f"Tensor {i} shape: {tensor.size()}") 
# # Tensor 0 shape: torch.Size([19427, 3])
# # Tensor 1 shape: torch.Size([19427, 3])
# # Tensor 2 shape: torch.Size([19427, 261])
# # Tensor 3 shape: torch.Size([19427, 2888])
# # Tensor 4 shape: torch.Size([19427, 26])
# # batch size , number of classes 
# ############################################

# # TRYING SOMETHING ELSE ON CAT COLUMNS THAT ARE LABEL ENCODED 
# ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # try this on Thursday 
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# encoded_data = encoder.fit_transform(df[['category']])
# categorical_tensor = torch.tensor(encoded_data, dtype=torch.float)
# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # Convert label-encoded categorical columns to tensors
# label_encoded_categorical_tensors = []
# for col in categorical_columns_list:
#     unique_values = df[col].unique()
#     tensor = torch.tensor([unique_values.tolist().index(value) for value in df[col]])
#     label_encoded_categorical_tensors.append(tensor)
# for i, tensor in enumerate(label_encoded_categorical_tensors):
#     print(f"Tensor {i} shape: {tensor.size()}")
# # Concatenate all tensors
# # input_tensor = torch.cat((numeric_tensor, *(tensor.t() for tensor in categorical_tensor), *label_encoded_categorical_tensors), dim=1)
# input_tensor = torch.cat((numeric_tensor, *(tensor.t() for tensor in categorical_tensor), *([tensor.unsqueeze(0) for tensor in label_encoded_categorical_tensors])), dim=1)
# # attempt 1 
# # tensorized_categorical_columns = []

# # # Iterate over the categorical columns
# # for col in categorical_columns_list:
# #     # Get the unique values in the column
# #     unique_values = df[col].unique()
    
# #     # Create a tensor for the column
# #     tensor = torch.tensor([unique_values.tolist().index(value) for value in df[col]])
    
# #     # Append the tensor to the list
# #     tensorized_categorical_columns.append(tensor)

# # torch.stack([torch.tensor([unique_values.tolist().index(value) for value in df[col]]) for col in categorical_columns_list])

# # input_tensor = torch.cat((numeric_tensor, *(tensor.t() for tensor in categorical_tensor), tensorized_categorical_columns), dim=1)
# ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # Concatenate numeric, categorical, and one-hot encoded tensors

# input_tensor = torch.cat((numeric_tensor, *(tensor.t() for tensor in categorical_tensor), *one_hot_encoded_tensors), dim=1)



# # check shape 
# input_tensor.shape 
# # torch.Size([19427, 3208])

# # turn target variable into tensors 
# target_data = torch.tensor(target_data, dtype=torch.long) 
# target_data.shape  
# # torch.Size([19427])
# ###########################################################################################################
# # step 3  - create a neural network 
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(3208, 128)  # input layer (3208) neurons -> hidden layer (128) neurons 
#         self.fc2 = nn.Linear(128, 64)   # hidden layer (128) -> hidden layer (64)
#         self.fc4 = nn.Linear(64, 10)  # hidden layer (64) -> hidden layer (10)
#         # allow the model to learn more complex representations between features and the target variable 
#         # improves the model performance 
#         self.fc3 = nn.Linear(64, 2)     # hidden layer (64) -> output layer (2)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))      # activation function for hidden layer
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001)


# #####################
# # Step 4 - train test split 
# # Split the data into training, validation, and test sets
# train_size = int(0.8 * len(input_tensor)) # 15541
# val_size = int(0.1 * len(input_tensor)) # 1942
# test_size = len(input_tensor) - train_size - val_size #1944
# #########
# train_index = torch.arange(train_size)
# # check shape 
# train_index.shape # torch.Size([15541])
# #########
# val_index = torch.arange(train_size, train_size + val_size)
# val_index.shape # torch.Size([1942])
# #########
# test_index = torch.arange(train_size + val_size, len(input_tensor))
# test_index.shape # torch.Size([1944])
# ########
# print(train_index.shape, val_index.shape, test_index.shape)
# #torch.Size([15541]) torch.Size([1942]) torch.Size([1944])

# ######################
# # Step 5  - extract data 

# train_input_tensor = input_tensor[train_index]
# train_target_data = target_data[train_index]
# val_input_tensor = input_tensor[val_index]
# val_target_data = target_data[val_index]
# test_input_tensor = input_tensor[test_index]
# test_target_data = target_data[test_index]

# print(train_input_tensor.shape, 
#       train_target_data.shape, 
#       val_input_tensor.shape,
#       val_target_data.shape,
#       test_input_tensor.shape,
#       test_target_data.shape )
# #torch.Size([15541, 3208]) torch.Size([15541]) torch.Size([1942, 3208]) torch.Size([1942]) torch.Size([1944, 3208]) torch.Size([1944])

# ######################
# # Step 6  - train the model 
# for epoch in range(10):
#     for i in range(len(train_input_tensor)):
#         inputs = train_input_tensor[i].unsqueeze(0)
#         labels = train_target_data[i].unsqueeze(0)
#         outputs = net(inputs)
#         print(outputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# ######################
# # Step 7  - validate the model 
# val_loss = 0
# correct = 0
# with torch.no_grad():
#     for i in range(len(val_input_tensor)):
#         inputs = val_input_tensor[i].unsqueeze(0)
#         labels = val_target_data[i].unsqueeze(0)
#         print(labels)
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         val_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labels).sum().item()
# accuracy = correct / len(val_target_data)
# print(f'Validation Loss: {val_loss / len(val_target_data)}')
# print(f'Validation Accuracy: {accuracy:.2f}%')

# ########################
# # Step 8   - Test the model 
# test_loss = 0
# correct = 0
# with torch.no_grad():
#     for i in range(len(test_input_tensor)):
#         inputs = test_input_tensor[i].unsqueeze(0)
#         print(inputs)
#         labels = test_target_data[i].unsqueeze(0)
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labels).sum().item()
# accuracy = correct / len(test_target_data)
# print(f'Test Loss: {test_loss / len(test_target_data)}')
# print(f'Test Accuracy: {accuracy:.2f}%')




# # TO ADJUST MODEL - adjust the hyper parameters to improve model performance 
# # learning rate 
# # number of epochs 












# ################## ALL CODE BELOW IS OLD 


# # define the max sequence length 
# # max_len = 100 # adjust if needed, based on data, i think 100 is fine. 


# # # Convert categorical columns to a tensor
# # categorical_tensor = [
# #     rnn.pad_sequence([
# #         torch.tensor([word_to_ix[col].get(word, 0) for word in re.sub(r'[^\w\s]', '', row).lower().split()])
# #         for row in df[col]
# #     ], batch_first=True, padding_value=0, max_length=max_len) for col in longer_cat_col_list
# # ]

# # Convert categorical columns to a tensor
# # categorical_tensor = [
# #     rnn.pad_sequence([
# #         torch.tensor([word_to_ix[col].get(word, 0) for word in re.sub(r'[^\w\s]', '', row).lower().split()])
# #         for row in df[col]
# #     ]) for col in longer_cat_col_list
# # ]

# # Convert one-hot encoded columns to a tensor
# # one_hot_encoded_columns_list = {}
# # for col in categorical_columns_list:
# #     if col not in longer_cat_col_list:
# #         unique_values = df[col].unique()
# #         one_hot_encoded_columns_list[col] = torch.tensor([[1 if value == v else 0 for v in unique_values] for value in df[col]])
# # # Convert the dictionary view object to a list of tensors
# # one_hot_encoded_tensors = list(one_hot_encoded_columns_list.values())

# # padded_one_hot_tensors = [
# #     rnn.pad_sequence(tensor, batch_first=True, padding_value=0, max_length=max_len)
# #     for tnesor in one_hot_encoded_tensors
# # ]
# # Stack the numeric, categorical, and simple categorical tensors
# # input_tensor = torch.cat((numeric_tensor, *categorical_tensor, *simple_categorical_tensors), dim=1)
# # input_tensor = torch.cat((numeric_tensor.unsqueeze(0), *categorical_tensor, *one_hot_encoded_tensors), dim=0)
# # Ensure all tensors have the same number of dimensions
# numeric_tensor = numeric_tensor.unsqueeze(0)
# categorical_tensors = [tensor.unsqueeze(0) for tensor in categorical_tensor]
# # one_hot_encoded_tensors = [tensor.unsqueeze(0) for tensor in one_hot_encoded_tensors]

# # Stack the numeric, categorical, and simple categorical tensors
# input_tensor = torch.cat((numeric_tensor, *categorical_tensors, *one_hot_encoded_tensors), dim=0)



# # # Convert simple categorical columns to tensors
# # simple_categorical_tensors = [
# #     torch.tensor([value_to_ix[value] for value in df[col]], dtype=torch.long)
# #     for col in categorical_columns_list
# #     if col not in longer_cat_col_list
# # ]













# # Split data into training and testing sets
# from sklearn.model_selection import train_test_split
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)










# from sklearn.model_selection import train_test_split


# # Train-test split
# train_df, test_df = train_test_split(df, test_size=0.2)
# train_df.shape # 15541, 29 
# test_df.shape # 3886, 29 

# train_dict = train_df.to_dict(orient='records') ## train_dict is now a list of dictionaries, one for each row.

# test_dict = test_df.to_dict(orient='records')


# # dict_df = df.to_dict(orient='records') # returns a list 
# # dict_df[0]
# #### from here up is good to go !!! ### data preprocessing is complete 


# # new 
# # class CustomDataset(Dataset):
# #     def __init__(self, dict_df, numeric_columns, simple_cat_columns, target_data):
# #         self.dict_df = dict_df  # Use the entire dict for flexibility
# #         self.numeric_columns = numeric_columns
# #         self.simple_cat_columns = simple_cat_columns
# #         self.target_data = target_data

# #     def __len__(self):
# #         return len(self.target_data)

# #     def __getitem__(self, idx):
# #         # Gather numeric data
# #         numeric_data = {col: self.dict_df[col][idx] for col in self.numeric_columns}
# #         numeric_data_tensor = torch.tensor([value for _, value in numeric_data.items()], dtype=torch.float32)
        
# #         # Gather simple categorical data
# #         simple_cat_data = {col: self.dict_df[col][idx] for col in self.simple_cat_columns}
# #         simple_cat_data_tensor = torch.tensor([value for _, value in simple_cat_data.items()], dtype=torch.long)  # Assuming they're already encoded as integers
        
# #         # Handle longer categorical variables separately as before
# #         observedPropertyDeterminandCode = torch.tensor(self.dict_df['observedPropertyDeterminandCode'][idx], dtype=torch.long)
# #         resultUom = torch.tensor(self.dict_df['resultUom'][idx], dtype=torch.long)
        
# #         # Return a combined dictionary
# #         return {
# #             'numeric_data': numeric_data_tensor,
# #             'simple_cat_data': simple_cat_data_tensor,
# #             'observedPropertyDeterminandCode': observedPropertyDeterminandCode,
# #             'resultUom': resultUom,
# #             'target': torch.tensor(self.target_data[idx], dtype=torch.float32)
# #         }

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data):
#         self.dict_df = dict_df
#         self.numeric_columns = numeric_columns
#         self.simple_cat_columns = simple_cat_columns
#         self.complex_cat_columns = complex_cat_columns  # complex_cat_columns is a dict with keys as column names and values as mappings
#         self.target_data = target_data

#     def __len__(self):
#         return len(self.target_data)

#     def __getitem__(self, idx):
#         # Handling numeric data
#         numeric_data = [self.dict_df[col][idx] for col in self.numeric_columns]
#         numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32)

#         # Handling simple categorical data (already encoded)
#         simple_cat_data = [self.dict_df[col][idx] for col in self.simple_cat_columns]
#         simple_cat_tensor = torch.tensor(simple_cat_data, dtype=torch.long)

#         # Handling complex categorical data (requiring embeddings)
#         complex_cat_tensors = []
#         for col, mapping in self.complex_cat_columns.items():
#             encoded_value = mapping[self.dict_df[col][idx]]  # Assume dict_df[col][idx] is directly mappable
#             complex_cat_tensors.append(torch.tensor(encoded_value, dtype=torch.long))

#         target = torch.tensor(self.target_data[idx], dtype=torch.float32)

#         return {
#             'numeric_data': numeric_tensor,
#             'simple_cat_data': simple_cat_tensor,
#             'complex_cat_data': torch.stack(complex_cat_tensors),  # Stack complex categorical tensors
#             'target': target
#         }
        

# # new 
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader

# def custom_collate_fn(batch):
#     numeric_data = torch.stack([item['numeric_data'] for item in batch])
#     simple_cat_data = torch.stack([item['simple_cat_data'] for item in batch])
#     complex_cat_data = [item['complex_cat_data'] for item in batch]
#     targets = torch.stack([item['target'] for item in batch])

#     # Padding the complex categorical data
#     # Assuming complex_cat_data is a list of tensors where each tensor can have a different length
#     padded_complex_cat_data = pad_sequence(complex_cat_data, batch_first=True, padding_value=0)
    
#     return {
#         'numeric_data': numeric_data,
#         'simple_cat_data': simple_cat_data,
#         'complex_cat_data': padded_complex_cat_data,
#         'target': targets
#     }

# dataset = CustomDataset(dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


# # train / test split 
# from sklearn.model_selection import train_test_split

# # Assuming your data is already loaded into variables: dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data

# # Splitting the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(dict_df, target_data, test_size=0.2, random_state=42)

# # Creating datasets for training and testing
# train_dataset = CustomDataset(X_train, numeric_columns, simple_cat_columns, complex_cat_columns, y_train)
# test_dataset = CustomDataset(X_test, numeric_columns, simple_cat_columns, complex_cat_columns, y_test)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)



# losses = []
# val_losses = []
# loss_function = nn.BCELoss()
# model = NGramLanguageModeler(len(speakers), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)










# ###################
# ###################
# ###################
# # original below 

# class CustomDataset(Dataset):
#     def __init__(self, dict_df, numeric_columns, target_data):
#         # Validation check: Ensure all columns have the same length
#         sample_length = len(dict_df[numeric_columns[0]])
#         for col in numeric_columns + ['observedPropertyDeterminandCode', 'resultUom']:
#             if len(dict_df[col]) != sample_length:
#                 raise ValueError(f"Column '{col}' length does not match other columns.")
        
#         # Assuming numeric data can be safely converted to a NumPy array
#         self.numeric_data = np.array([dict_df[col] for col in numeric_columns], dtype=np.float32).T
#         # Assuming target_data is passed as a list or numpy array
#         self.target_data = np.array(target_data, dtype=np.float32)
#         # Store tokenized sequences as lists of integers for dynamic padding
#         self.observedPropertyDeterminandCode = dict_df['observedPropertyDeterminandCode']
#         self.resultUom = dict_df['resultUom']
        
#     def __len__(self):
#         return len(self.target_data)
    
#     def __getitem__(self, idx):
#         # Return numeric data, tokenized sequences as lists, and target for each item
#         return {
#             'numeric_data': torch.tensor(self.numeric_data[idx], dtype=torch.float),
#             'observedPropertyDeterminandCode': self.observedPropertyDeterminandCode[idx],
#             'resultUom': self.resultUom[idx],
#             'target': torch.tensor(self.target_data[idx], dtype=torch.float)
#         }

# from torch.nn.utils.rnn import pad_sequence

# # pad for those those categorical variables that require padding 
# def collate_fn(batch):
#     numeric_data = torch.stack([item['numeric_data'] for item in batch])
#     targets = torch.tensor([item['target'] for item in batch], dtype=torch.float)

#     observedPropertyDeterminandCode = pad_sequence([torch.tensor(item, dtype=torch.long) for item in [b['observedPropertyDeterminandCode'] for b in batch]], batch_first=True, padding_value=0)
#     resultUom = pad_sequence([torch.tensor(item, dtype=torch.long) for item in [b['resultUom'] for b in batch]], batch_first=True, padding_value=0)

#     # Return a single batch object
#     return {
#         'numeric_data': numeric_data,
#         'observedPropertyDeterminandCode': observedPropertyDeterminandCode,
#         'resultUom': resultUom,
#         'targets': targets
#     }



# # Assuming target_data is your target variable list or numpy array
# custom_dataset = CustomDataset(dict_df, numeric_columns, target_data)

# # DataLoader remains unchanged
# dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# # dataloader = DataLoader(CustomDataset(dict_df, numeric_columns, target_column), batch_size=32, shuffle=True, collate_fn=collate_fn)


# ############### here TUESDAY good from here up !!! #############
# # Start here Wednesday 
# # model suggestion - 4/2
# # class MyModel(nn.Module):
# #     def __init__(self, num_numeric_features, embedding_sizes):
# #         super(MyModel, self).__init__()
# #         self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
# #         # Assuming one embedding layer per categorical variable, `embedding_sizes` is a list of tuples:
# #         # [(num_categories_variable_1, embedding_dim_1), ..., (num_categories_variable_n, embedding_dim_n)]
        
# #         # Example for numeric and embedding combination
# #         self.fc_layers = nn.Sequential(
# #             nn.Linear(num_numeric_features + sum([size for _, size in embedding_sizes]), 128),  # Adjust sizes accordingly
# #             nn.ReLU(),
# #             nn.Linear(128, 1),
# #             nn.Sigmoid()
# #         )
        
# #     def forward(self, x_numeric, x_categorical):
# #         embeddings = [embedding(cat) for embedding, cat in zip(self.embeddings, x_categorical)]
# #         x = torch.cat([x_numeric] + embeddings, dim=1)
# #         return self.fc_layers(x)
# class MyModel(nn.Module):
#     def __init__(self, num_numeric_features, num_categories_list):
#         super(MyModel, self).__init__()
#         self.embedding_dim = 10  # Set the embedding dimension
        
#         # Initialize embeddings with embedding_dim for each categorical variable
#         self.embeddings = nn.ModuleList([nn.Embedding(num_categories, self.embedding_dim) for num_categories in num_categories_list])
        
#         # Calculate the total size of concatenated features: numeric features + all embeddings
#         total_feature_size = num_numeric_features + len(num_categories_list) * self.embedding_dim
        
#         # Define the fully connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(total_feature_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )



#     def forward(self, x_numeric, x_categorical):
#             # Embedding categorical variables and reshaping
#             cat_embeddings = []
#             for i, cat_var in enumerate(x_categorical):
#                 embedded = self.embeddings[i](cat_var)
#                 cat_embeddings.append(embedded)
            
#             # Flatten the embeddings and concatenate with numeric features
#             cat_embeddings = torch.cat(cat_embeddings, dim=-1)
#             combined = torch.cat([x_numeric, cat_embeddings], dim=1)
            
#             # Pass through fully connected layers
#             return self.fc_layers(combined)
    
    
#     # def forward(self, x_numeric, x_categorical):
#     #     # Embed categorical variables and concatenate with numeric features
#     #     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
#     #     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
#     #     return self.fc_layers(x)


# # def forward(self, x_numeric, x_categorical):
# #     # Diagnostics: Check max index for each categorical variable
# #     for i, cat_var in enumerate(x_categorical):
# #         max_index = torch.max(cat_var)
# #         if max_index >= self.embeddings[i].num_embeddings:
# #             raise ValueError(f"Out of range index {max_index} for embedding layer {i} with size {self.embeddings[i].num_embeddings}")
    
# #     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
# #     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
# #     return self.fc_layers(x)

# # def forward(self, x_numeric, x_categorical):
# #     for i, (embedding, cat_var) in enumerate(zip(self.embeddings, x_categorical)):
# #         max_index = torch.max(cat_var)
# #         print(f"Max index for embedding {i}: {max_index.item()} (Size: {embedding.num_embeddings})")
# #         assert max_index < embedding.num_embeddings, f"Index out of range for embedding {i}"

# #     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
# #     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
# #     return self.fc_layers(x)


# def calculate_accuracy(y_true, y_pred):
#     predicted = y_pred.ge(.5).view(-1)  # Using 0.5 as threshold
#     return (y_true == predicted).sum().float() / len(y_true)



# # Train/Test Split 
# from sklearn.model_selection import train_test_split

# # Convert dict_df back to DataFrame for easy manipulation
# df = pd.DataFrame(dict_df)
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(df, target_data, test_size=0.2, random_state=42)

# # Convert splits back to the required format if necessary
# train_dict_df = X_train.to_dict('list')
# test_dict_df = X_test.to_dict('list')

# # Initialize custom datasets
# train_dataset = CustomDataset(train_dict_df, numeric_columns, y_train)
# test_dataset = CustomDataset(test_dict_df, numeric_columns, y_test)

# # Initialize DataLoaders
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)



# # Initialize MyModel with the appropriate number of numeric features and a list of the number of unique categories per categorical variable
# model = MyModel(num_numeric_features=3, num_categories_list=[10, 20])  # Adjust these values based on your actual data
# loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Modify the training loop
# for epoch in range(100):
#     model.train()
#     total_loss, total_accuracy = 0, 0
    
#     for batch in train_dataloader:
#         # Assuming this unpacking matches your data structure
#         x_numeric, x_categorical, targets = batch['numeric_data'], [batch['observedPropertyDeterminandCode'], batch['resultUom']], batch['targets']
        
#         optimizer.zero_grad()
#         predictions = model(x_numeric, x_categorical)
#         loss = loss_function(predictions.squeeze(), targets)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         total_accuracy += calculate_accuracy(targets, predictions.squeeze()).item()

#     # Calculate average loss and accuracy
#     avg_loss = total_loss / len(train_dataloader)
#     avg_accuracy = total_accuracy / len(train_dataloader)
#     print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}')

#     # Evaluate on test data
#     model.eval()
#     with torch.no_grad():
#         test_accuracy = sum(calculate_accuracy(batch['targets'], model(batch['numeric_data'], [batch['cat_var1'], batch['cat_var2']]).squeeze()).item() for batch in test_dataloader) / len(test_dataloader)
#     print(f'Test Accuracy: {test_accuracy:.4f}')

# ###############










# # Assuming `data_loader` is your DataLoader instance providing batches in the correct format
# # for epoch in range(100):
# #     total_loss = 0
# #     for batch in data_loader:
# #         x_numeric, x_categorical, targets = batch['numeric_data'], [batch['cat_var1'], batch['cat_var2']], batch['targets']  # Adapt as necessary

# #         model.zero_grad()
# #         predictions = model(x_numeric, x_categorical)
# #         loss = loss_function(predictions, targets.unsqueeze(1))  # Ensure target shape matches prediction
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #     print(f'Epoch {epoch}, Loss: {total_loss}')


















# ##################################Thursday
# # prepping data and adding in padding V1 

# # class CustomDataset(Dataset):
# #     def __init__(self, dict_df, numeric_columns, target_column):
# #         # Assuming numeric data can be safely converted to a NumPy array (check if this assumption holds true)
# #         self.numeric_data = np.array([dict_df[col] for col in numeric_columns], dtype=np.float32).T  # Transpose to align with the number of samples
# #         self.target_data = np.array(dict_df[target_column], dtype=np.float32)
        
# #         # Store tokenized sequences as lists of integers for dynamic padding
# #         self.observedPropertyDeterminandCode = dict_df['observedPropertyDeterminandCode']
# #         self.resultUom = dict_df['resultUom']
        
# #     def __len__(self):
# #         return len(self.target_data)
    
# #     def __getitem__(self, idx):
# #         # Return numeric data, tokenized sequences as lists, and target for each item
# #         return {
# #             'numeric_data': torch.tensor(self.numeric_data[idx], dtype=torch.float),
# #             'observedPropertyDeterminandCode': self.observedPropertyDeterminandCode[idx],
# #             'resultUom': self.resultUom[idx],
# #             'target': torch.tensor(self.target_data[idx], dtype=torch.float)
# #         }

# # ###########
# # # Dynamic Padding Sequence 
# # from torch.nn.utils.rnn import pad_sequence

# # def collate_fn(batch):
# #     numeric_data = torch.stack([item['numeric_data'] for item in batch])
# #     targets = torch.tensor([item['target'] for item in batch], dtype=torch.float)

# #     # Dynamic padding for your multi-word categorical columns
# #     observedPropertyDeterminandCode = pad_sequence([torch.tensor(item['observedPropertyDeterminandCode']) for item in batch], batch_first=True, padding_value=0)
# #     resultUom = pad_sequence([torch.tensor(item['resultUom']) for item in batch], batch_first=True, padding_value=0)

# #     return numeric_data, observedPropertyDeterminandCode, resultUom, targets

# # # Adjusted DataLoader initialization without 'mappings' argument
# # dataloader = DataLoader(CustomDataset(dict_df, numeric_columns, target_column), batch_size=32, shuffle=True, collate_fn=collate_fn)

# ################ here up okay i think, now need to figure out the model ##### 

# # model suggestion - 4/2
# class MyModel(nn.Module):
#     def __init__(self, num_numeric_features, embedding_sizes):
#         super(MyModel, self).__init__()
#         self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
#         # Assuming one embedding layer per categorical variable, `embedding_sizes` is a list of tuples:
#         # [(num_categories_variable_1, embedding_dim_1), ..., (num_categories_variable_n, embedding_dim_n)]
        
#         # Example for numeric and embedding combination
#         self.fc_layers = nn.Sequential(
#             nn.Linear(num_numeric_features + sum([size for _, size in embedding_sizes]), 128),  # Adjust sizes accordingly
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x_numeric, x_categorical):
#         embeddings = [embedding(cat) for embedding, cat in zip(self.embeddings, x_categorical)]
#         x = torch.cat([x_numeric] + embeddings, dim=1)
#         return self.fc_layers(x)





# # Start below here on MOnday 
# # getting the following error when running the below RuntimeError: Tensors must have same number of dimensions: got 2 and 3

# #############
# # NN MODEL 
# import torch.nn.functional as F

# class CustomModel(nn.Module):
#     def __init__(self, numeric_size, embedding_dims, vocab_sizes):
#         super(CustomModel, self).__init__()
#         # Embedding layers for categorical data
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) 
#             for vocab_size, embedding_dim in zip(vocab_sizes, embedding_dims)
#         ])
#         # Linear layer for numeric data
#         self.numeric_layer = nn.Linear(numeric_size, 128)
#         # Final layer after concatenating embeddings and numeric features
#         self.final_layer = nn.Linear(128 + sum(embedding_dims), 1)

#     def forward(self, numeric_data, observedPropertyDeterminandCode, resultUom):
#         cat_embeddings = torch.cat([self.embeddings[0](observedPropertyDeterminandCode), 
#                                     self.embeddings[1](resultUom)], dim=1)
#         numeric_features = F.relu(self.numeric_layer(numeric_data))
#         combined_features = torch.cat((numeric_features, cat_embeddings), dim=1)
#         output = torch.sigmoid(self.final_layer(combined_features))
#         return output


# ##################################working the padding section above 
# # since we are using collate_fn we dont need to use context size 

# # Initialize the model, loss function, and optimizer
# # Corrected model initialization without context_sizes
# model = CustomModel(numeric_size=len(numeric_columns), embedding_dims=[10, 10], vocab_sizes=[len(vocab1), len(vocab2)])
# NUM_EPOCHS = 10  # Define the number of epochs for training


# ########## adjust here if needed ####
# loss_function = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001) #Optimizer Alternatives: If after some initial experiments you find that SGD isn't performing well, consider switching to optim.Adam for its adaptive learning rate capabilities:

# # Training loop
# # Assume NUM_EPOCHS is defined
# for epoch in range(NUM_EPOCHS):
#     for batch in dataloader:
#         numeric_data, observedPropertyDeterminandCode, resultUom, targets = batch
#         model.zero_grad()
#         # Adjust the model call according to its forward method's expected arguments
#         output = model(numeric_data, observedPropertyDeterminandCode, resultUom)
#         loss = loss_function(output.squeeze(), targets)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# #################### working here up ##########










# #########DATAFRAME VERSION ##########BELOW WED. 

# ######################## Prep data ########################
# # look at data types 
# df.dtypes
# # identify numeric columns 
# numeric_columns = df.select_dtypes('number').columns
# # identify categorical columns 
# categorical_columns = df.select_dtypes('object').columns
# # look at the cardinality of the categorical variables
# df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')

# df['procedureAnalysedMedia'].head() # this will be the target variable 
# # label encode the 2 categories (water (1) sediment (0))
# target_encoder = LabelEncoder()
# df['procedureAnalysedMedia'] = target_encoder.fit_transform(df['procedureAnalysedMedia'])
# target = df['procedureAnalysedMedia'] # target variable 
# df['procedureAnalysedMedia'].head() # water selected as 1 water, 0 sediment
# # maybe we dont drop the label -- we didnt in the example by Santerre 
# df.drop('procedureAnalysedMedia', axis=1, inplace=True) # drop our target variable from the df 
# df.columns # check 

# ##########################
# # categorical columns identified and put into a list 
# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# # Define the embedding layer for the numeric columns
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# target_column = 'procedureAnalysedMedia'

# #
# # Assuming df is your DataFrame
# # Example preprocessing for numeric columns
# scaler = StandardScaler()
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])



# ##### here start down now, up above okay #####
# # columns observedPropertyDeterminandCode , resultUom to be treated like they need to be word embedded 


# # Convert categorical columns to 'category' dtype
# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# for col in categorical_columns:
#     df[col] = df[col].astype('category')








# #############################working here down ############
# # Attempt 2 
# # Tokenize and encode categorical columns with sequences of words
# CONTEXT_SIZE = 3  # Define a fixed context size for simplicity
# def tokenize_and_encode(column, context_size=CONTEXT_SIZE):
#     vocab = set(word for row in df[column].dropna() for word in row.split())
#     word_to_ix = {word: i+1 for i, word in enumerate(vocab)}  # Start indexing from 1 to reserve 0 for padding
    
#     # Tokenize and adjust sequences according to the context size
#     encoded_sequences = []
#     for row in df[column].fillna(''):
#         words = row.split()
#         encoded_sequence = [word_to_ix[word] if word in word_to_ix else 0 for word in words[:context_size]]
#         encoded_sequence += [0] * (context_size - len(encoded_sequence))  # Padding
#         encoded_sequences.append(encoded_sequence)
        
#     return encoded_sequences, len(vocab) + 1  # +1 to account for padding token

# encoded_observedPropertyDeterminandCode, vocab_size_observedPropertyDeterminandCode = tokenize_and_encode('observedPropertyDeterminandCode')
# encoded_resultUom, vocab_size_resultUom = tokenize_and_encode('resultUom')

# # Add encoded sequences back to DataFrame
# df['encoded_observedPropertyDeterminandCode'] = encoded_observedPropertyDeterminandCode
# df['encoded_resultUom'] = encoded_resultUom

# # Update categorical columns list to include only the encoded versions
# categorical_columns = ['encoded_observedPropertyDeterminandCode', 'encoded_resultUom']

# class CustomDataset(Dataset):
#     def __init__(self, dataframe, target_column, numeric_columns, categorical_columns):
#         self.numeric_data = dataframe[numeric_columns].values.astype(np.float32)
#         self.categorical_data = np.array(dataframe[categorical_columns].tolist()).astype(np.int64)
#         self.targets = dataframe[target_column].values.astype(np.float32)
        
#     def __len__(self):
#         return len(self.targets)
    
#     def __getitem__(self, idx):
#         return {
#             'numeric_data': torch.tensor(self.numeric_data[idx], dtype=torch.float),
#             'categorical_data': torch.tensor(self.categorical_data[idx], dtype=torch.long),
#             'target': torch.tensor(self.targets[idx], dtype=torch.float)
#         }
# class CustomModel(nn.Module):
#     def __init__(self, numeric_size, embedding_sizes, vocab_sizes):
#         super(CustomModel, self).__init__()
#         self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for vocab_size, embedding_size in zip(vocab_sizes, embedding_sizes)])
#         total_embedding_size = sum(embedding_sizes)
        
#         self.linear1 = nn.Linear(total_embedding_size + numeric_size, 128)
#         self.linear2 = nn.Linear(128, 1)

#     def forward(self, numeric_data, categorical_data):
#         cat_embeddings = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.embeddings)]
#         cat_embeddings_combined = torch.cat(cat_embeddings, dim=1)
#         combined = torch.cat([cat_embeddings_combined, numeric_data], dim=1)
        
#         out = F.relu(self.linear1(combined))
#         out = torch.sigmoid(self.linear2(out))
#         return out

# # Example embedding sizes for each categorical column (adjust as needed)
# embedding_sizes = [10, 10]  # Example: 10-dimensional embeddings for both columns
# vocab_sizes = [vocab_size_observedPropertyDeterminandCode, vocab_size_resultUom]

# model = CustomModel(len(numeric_columns), embedding_sizes, vocab_sizes)
# # Initialize your dataset and data loader
# dataset = CustomDataset(df, 'procedureAnalysedMedia', numeric_columns, categorical_columns)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Define loss function and optimizer
# loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop (simplified)
# for epoch in range(10):  # Example: 10 epochs
#     for batch in dataloader:
#         optimizer.zero_grad()
#         output = model(batch['numeric_data'], batch['categorical_data'])
#         loss = loss_function(output.squeeze(), batch['target'])
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
























