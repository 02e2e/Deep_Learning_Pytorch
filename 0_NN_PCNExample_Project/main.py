# Mock up of PCN Sample - EDA, Word Embedding, RF, NN 

import csv

def load_csv(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data = [row for row in csvreader]
        headers = csvreader.fieldnames
        return data, headers

def identify_data_types(data):
    data_types = {}
    for key in data[0].keys():
        # Assume numerical data type by default
        data_types[key] = 'numerical'
        for row in data:
            value = row[key]
            if value and not value.isdigit():  # If value is not empty and not entirely digits
                data_types[key] = 'categorical'
                break
    return data_types

def check_missing_values(data):
    missing_values = {header: 0 for header in data[0].keys()}
    for row in data:
        for key, value in row.items():
            if not value.strip():  # Check for empty or whitespace-only values
                missing_values[key] += 1
    return missing_values

def check_integer_punctuation(data):
    punctuation_issues = {}
    for key in data[0].keys():
        punctuation_issues[key] = any(',' in row[key] or '.' in row[key] for row in data if row[key].isdigit())
    return punctuation_issues

# Load CSV file
file_path = 'CleanDF.csv'  # Replace with your file path
data, headers = load_csv(file_path)

# Identify data types
data_types = identify_data_types(data)
print("Data Types:")
print(data_types)

# Check for missing values
missing_values = check_missing_values(data)
print("\nMissing Values:")
print(missing_values)

# Check for punctuation in integer data
punctuation_issues = check_integer_punctuation(data)
print("\nInteger Data Punctuation Issues:")
print(punctuation_issues)

print(data[0])




import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, TextVectorization

# Data Preparation
def prepare_data_for_word_embeddings(data_df, embedding_dim=10):
    category_features = data_df.select_dtypes('category').columns
    numeric_features = data_df.select_dtypes('number').columns

    # Ensure Text Compatibility
    for col in category_features:
        data_df[col] = data_df[col].astype(str)

    text_vectorizer = TextVectorization(output_mode='int', output_sequence_length=1)  
    text_vectorizer.adapt(data_df[category_features].to_numpy())

    embedding_layers = {}
    for col in category_features:
       embedding_layers[col] = Embedding(input_dim=text_vectorizer.vocabulary_size(), 
                                         output_dim=embedding_dim)

    return data_df, category_features, numeric_features, text_vectorizer, embedding_layers

# Model Building Helper
def create_embedding_input(feature_name, vocab_size, embedding_dim):
    inpt = tf.keras.layers.Input(shape=(1,), name='input_' + feature_name) 
    embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inpt)
    embed_reshaped = tf.keras.layers.Reshape(target_shape=(embedding_dim,))(embed)
    return inpt, embed_reshaped

# --- Example Usage ---
# Assuming you have your 'df_clean' DataFrame ready

df_clean, category_features, numeric_features, text_vectorizer, embedding_layers = prepare_data_for_word_embeddings(df_clean.copy(), embedding_dim=10)

models = []
inputs = []
for cat in categorical_features:
    vocab_size = text_vectorizer.vocabulary_size()
    inputs, models = create_embedding_input(cat, vocab_size, embedding_dim)

# ... (Rest of your model architecture, like handling the numeric_features,
#      combining embeddings, adding dense layers, defining outputs, etc.) 
for cat in categorical_features:
    vocab_size = data[cat].nunique()
    inpt = tf.keras.layers.Input(shape=(1,),\
                                 name='input_' + '_'.join(\
                                 cat.split(' ')))
    embed = tf.keras.layers.Embedding(vocab_size, 200,trainable=True,\
                                      embeddings_initializer=tf.initializers\
                                      .random_normal)(inpt)
    embed_rehsaped = tf.keras.layers.Reshape(target_shape=(200,))(embed)
    models.append(embed_rehsaped)
    inputs.append(inpt)
    
    
    
    
    
    #####
    
    import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Prepare Data
# Load your DataFrame with columns 'animal' and 'flower'
df = pd.read_csv("your_data.csv")

# Step 2: Preprocess Data
# Extract unique animals and flowers from the DataFrame
animals = set(df['animal'])
flowers = set(df['flower'])

# Step 3: Define the Model
# Define the embeddings for animals and flowers
animal_embeddings = nn.Embedding(len(animals), EMBEDDING_DIM)
flower_embeddings = nn.Embedding(len(flowers), EMBEDDING_DIM)

# Step 4: Train the Model
# You need to prepare your data for training, which involves encoding animal and flower columns into indices
# Then you can train the model similar to the original code, by iterating over epochs and updating parameters

# Step 5: Apply Embeddings
# After training, you can replace 'animal' and 'flower' columns with their corresponding embeddings

# For example:
# Replace 'animal' column with its embeddings
animal_indices = ...  # Encode animal data into indices
df['animal_embedding'] = animal_embeddings(torch.tensor(animal_indices))

# Replace 'flower' column with its embeddings
flower_indices = ...  # Encode flower data into indices
df['flower_embedding'] = flower_embeddings(torch.tensor(flower_indices))

# Remember to replace "your_data.csv" with the path to your dataset file,
# and replace ... with your code for encoding animal and flower data into indices.



import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Sample DataFrame with categorical and numeric columns
# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

# Extract unique values for each categorical column
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# Create word-to-index mapping for each categorical column
word_to_ix = {col: {word: i for i, word in enumerate(vocabularies[col])} for col in categorical_columns}

# Define the model
class CategoricalEmbedder(nn.Module):
    def __init__(self, vocabularies, embedding_dim):
        super(CategoricalEmbedder, self).__init__()
        self.embeddings = nn.ModuleDict({col: nn.Embedding(len(vocabularies[col]), embedding_dim) for col in vocabularies})
    
    def forward(self, inputs):
        embedded_inputs = [self.embeddings[col](input_) for col, input_ in inputs.items()]
        return torch.cat(embedded_inputs, dim=1)

# Train the model
model = CategoricalEmbedder(vocabularies, EMBEDDING_DIM)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Assuming you have prepared your data for training
# and have encoded the categorical columns into indices

# Now iterate over the data with a sliding window of size CONTEXT_SIZE
for i in range(CONTEXT_SIZE, len(df) - CONTEXT_SIZE):
    # Extract the context words for each categorical column
    context_idxs = {col: [] for col in categorical_columns}
    for j in range(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1):
        for col in categorical_columns:
            context_idxs[col].append(torch.tensor(word_to_ix[col][df.iloc[j][col]], dtype=torch.long))
    
    # Zero the gradients, forward pass, compute loss, backward pass, update weights
    model.zero_grad()
    embeddings = model(context_idxs)
    # Your training logic here



#######################
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Sample DataFrame with categorical and numeric columns
# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

# Extract unique values for each categorical column
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# Create word-to-index mapping for each categorical column
word_to_ix = {col: {word: i for i, word in enumerate(vocabularies[col])} for col in categorical_columns}

# Define the embedding layers for each categorical column
embeddings = {col: nn.Embedding(len(vocabularies[col]), EMBEDDING_DIM) for col in vocabularies}

# Define the model
class NGramLanguageModeler(nn.Module):
    def __init__(self, speaker_size, embedding_dim, context_size, embeddings):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.ModuleDict(embeddings)
        self.embeddings2 = nn.Embedding(speaker_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim+(context_size * embedding_dim), 128)
        self.linear2 = nn.Linear(128, 1)
    
    def forward(self, inputs):
        speaker, sentence = inputs
        sentence_embed = torch.cat([self.embeddings[col](sentence[col]) for col in sentence], dim=1)
        speaker_embed = self.embeddings2(speaker).view((1, -1))
        embeds_full = torch.cat((speaker_embed, sentence_embed), -1) 
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs

# Train the model
loss_function = nn.BCELoss()
model = NGramLanguageModeler(len(df['speaker'].unique()), EMBEDDING_DIM, CONTEXT_SIZE, embeddings)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Assuming you have prepared your data for training
# and have encoded the categorical columns into indices

# Now iterate over the data with a sliding window of size CONTEXT_SIZE
for i in range(CONTEXT_SIZE, len(df) - CONTEXT_SIZE):
    # Extract the context words and speaker index for each sample
    context_idxs = []
    for j in range(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1):
        word_idx = {col: torch.tensor(word_to_ix[col][df.iloc[j][col]], dtype=torch.long) for col in categorical_columns}
        speaker_idx = torch.tensor(speaker_to_ix[df.iloc[j]['speaker']], dtype=torch.long)
        context_idxs.append((speaker_idx, word_idx))
    
    # Zero the gradients, forward pass, compute loss, backward pass, update weights
    optimizer.zero_grad()
    log_probs = model(context_idxs)
    loss = loss_function(log_probs, torch.tensor([df.iloc[i]['target']], dtype=torch.float).resize_((1, 1)))
    loss.backward()
    optimizer.step()
    # Your training logic here




###########################
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample DataFrame with categorical and numeric columns
# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

# Define categorical columns and their respective sets of unique values
categorical_columns = ["animal", "flower", "car", "business"]
vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# Create word-to-index mapping and embeddings for each categorical column
word_to_ix = {}
embeddings = {}
for col in categorical_columns:
    word_to_ix[col] = {word: i for i, word in enumerate(vocabularies[col])}
    embeddings[col] = nn.Embedding(len(vocabularies[col]), EMBEDDING_DIM)

# Define the embedding layer for the numeric columns
numeric_embeddings = nn.ModuleDict({col: nn.Linear(1, EMBEDDING_DIM) for col in numeric_columns})

# Define the model
class CategoricalEmbedder(nn.Module):
    def __init__(self, embeddings, numeric_embeddings):
        super(CategoricalEmbedder, self).__init__()
        self.embeddings = nn.ModuleDict(embeddings)
        self.numeric_embeddings = nn.ModuleDict(numeric_embeddings)
        self.linear1 = nn.Linear(len(categorical_columns) * EMBEDDING_DIM + len(numeric_columns) * EMBEDDING_DIM, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        categorical_embeds = torch.cat([self.embeddings[col](inputs[col]) for col in inputs if col in categorical_columns], dim=1)
        numeric_embeds = torch.cat([self.numeric_embeddings[col](inputs[col]) for col in inputs if col in numeric_columns], dim=1)
        combined = torch.cat([categorical_embeds, numeric_embeds], dim=1)
        out = F.relu(self.linear1(combined))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs

# Train the model
loss_function = nn.BCELoss()
model = CategoricalEmbedder(embeddings, numeric_embeddings)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Assuming you have prepared your data for training
# and have encoded the categorical columns into indices

# Now iterate over the data with a sliding window of size CONTEXT_SIZE
for i in range(CONTEXT_SIZE, len(df) - CONTEXT_SIZE):
    # Extract the context words and numeric values for each sample
    context_idxs = {col: torch.tensor(word_to_ix[col][df.iloc[i][col]], dtype=torch.long) for col in categorical_columns}
    numeric_values = {col: torch.tensor(df.iloc[i][col], dtype=torch.float) for col in numeric_columns}
    
    # Zero the gradients, forward pass, compute loss, backward pass, update weights
    optimizer.zero_grad()
    log_probs = model({**context_idxs, **numeric_values})
    loss = loss_function(log_probs, torch.tensor([df.iloc[i]['target']], dtype=torch.float).resize_((1, 1)))
    loss.backward()
    optimizer.step()
    # Your training logic here
########################


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample DataFrame with categorical and numeric columns
# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

# Define constants
CONTEXT_SIZE = 2  # Define your desired context size
EMBEDDING_DIM = 10  # Define your desired embedding dimension

# Define categorical columns and their respective sets of unique values
categorical_columns = ["animal", "flower", "car", "business"]
vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# Create word-to-index mapping and embeddings for each categorical column
word_to_ix = {}
embeddings = {}
for col in categorical_columns:
    word_to_ix[col] = {word: i for i, word in enumerate(vocabularies[col])}
    embeddings[col] = nn.Embedding(len(vocabularies[col]), EMBEDDING_DIM)

# Define the embedding layer for the numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_embeddings = nn.ModuleDict({col: nn.Linear(1, EMBEDDING_DIM) for col in numeric_columns})

# Define the model
class CategoricalEmbedder(nn.Module):
    def __init__(self, vocabularies, numeric_embeddings, embedding_dim, context_size):
        super(CategoricalEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.embeddings = nn.ModuleDict({col: nn.Embedding(len(vocabularies[col]), embedding_dim) for col in vocabularies})
        self.numeric_embeddings = nn.ModuleDict(numeric_embeddings)
        self.linear1 = nn.Linear(len(categorical_columns) * embedding_dim + len(numeric_columns) * embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        categorical_embeds = torch.cat([self.embeddings[col](inputs[col]) for col in inputs if col in categorical_columns], dim=1)
        numeric_embeds = torch.cat([self.numeric_embeddings[col](inputs[col]) for col in inputs if col in numeric_columns], dim=1)
        combined = torch.cat([categorical_embeds, numeric_embeds], dim=1)
        out = F.relu(self.linear1(combined))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs

# Train the model
loss_function = nn.BCELoss()
model = CategoricalEmbedder(vocabularies, numeric_embeddings, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Assuming you have prepared your data for training
# and have encoded the categorical columns into indices

# Now iterate over the data with a sliding window of size CONTEXT_SIZE
for i in range(CONTEXT_SIZE, len(df) - CONTEXT_SIZE):
    # Extract the context words and numeric values for each sample
    context_idxs = {col: torch.tensor(word_to_ix[col][df.iloc[i][col]], dtype=torch.long) for col in categorical_columns}
    numeric_values = {col: torch.tensor(df.iloc[i][col], dtype=torch.float) for col in numeric_columns}
    
    # Zero the gradients, forward pass, compute loss, backward pass, update weights
    optimizer.zero_grad()
    log_probs = model({**context_idxs, **numeric_values})
    loss = loss_function(log_probs, torch.tensor([df.iloc[i]['target']], dtype=torch.float).resize_((1, 1)))
    loss.backward()
    optimizer.step()
    # Your training logic here
#####################



import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample DataFrame with categorical and numeric columns
# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

# Define constants
CONTEXT_SIZE = 2  # Define your desired context size
EMBEDDING_DIM = 10  # Define your desired embedding dimension

# Define categorical columns and their respective sets of unique values
categorical_columns = ["animal", "flower", "car", "business"]
vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# Create word-to-index mapping and embeddings for each categorical column
word_to_ix = {}
embeddings = {}
for col in categorical_columns:
    word_to_ix[col] = {word: i for i, word in enumerate(vocabularies[col])}
    embeddings[col] = nn.Embedding(len(vocabularies[col]), EMBEDDING_DIM)

# Define the embedding layer for the numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_embeddings = nn.ModuleDict({col: nn.Linear(1, EMBEDDING_DIM) for col in numeric_columns})

# Define the model
class CategoricalEmbedder(nn.Module):
    def __init__(self, embeddings, numeric_embeddings):
        super(CategoricalEmbedder, self).__init__()
        self.embeddings = nn.ModuleDict(embeddings)
        self.numeric_embeddings = nn.ModuleDict(numeric_embeddings)
        self.linear1 = nn.Linear(len(categorical_columns) * EMBEDDING_DIM + len(numeric_columns) * EMBEDDING_DIM, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        categorical_embeds = torch.cat([self.embeddings[col](inputs[col]) for col in inputs if col in categorical_columns], dim=1)
        numeric_embeds = torch.cat([self.numeric_embeddings[col](inputs[col]) for col in inputs if col in numeric_columns], dim=1)
        combined = torch.cat([categorical_embeds, numeric_embeds], dim=1)
        out = F.relu(self.linear1(combined))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs


# ###  ### TIM YOU ARE HERE START HERE FRIDAY 
losses = []
val_losses = []
model = CategoricalEmbedder(embeddings, numeric_embeddings)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Splitting the dataframe into training and validation sets
train_size = int(0.8 * len(df))
train_df, val_df = df[:train_size], df[train_size:]

for epoch in range(100):
    total_loss = 0
    for i, row in train_df.iterrows():
        # Prepare inputs for the training set
        context_idxs = {col: torch.tensor([word_to_ix[col][row[col]]], dtype=torch.long) for col in categorical_columns}
        numeric_values = {col: torch.tensor([row[col]], dtype=torch.float) for col in numeric_columns}
        
        # Zero the gradients, forward pass, compute loss, backward pass, update weights
        optimizer.zero_grad()
        log_probs = model({**context_idxs, **numeric_values})
        target = torch.tensor([row['target']], dtype=torch.float).resize_((1, 1))
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(train_df))
    model.eval()
    
    

    with torch.no_grad():
        total_val_loss = 0
        for i, row in val_df.iterrows():  # Using val_df for validation
            context_idxs = {col: torch.tensor([word_to_ix[col][row[col]]], dtype=torch.long) for col in categorical_columns}
            numeric_values = {col: torch.tensor([row[col]], dtype=torch.float) for col in numeric_columns}
            log_probs = model({**context_idxs, **numeric_values})
            val_target = torch.tensor([row['target']], dtype=torch.float).resize_((1, 1))
            val_loss = loss_function(log_probs, val_target)
            total_val_loss += val_loss.item()
        val_losses.append(total_val_loss / len(val_df))  # Using val_df for validation
    print('Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, losses[-1], val_losses[-1]))


# losses = []
# val_losses = []
# model = CategoricalEmbedder(embeddings, numeric_embeddings)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# for epoch in range(100):
#     total_loss = 0
#     for i, row in df.iterrows():
#         # Prepare inputs
#         context_idxs = {col: torch.tensor([word_to_ix[col][row[col]]], dtype=torch.long) for col in categorical_columns}
#         numeric_values = {col: torch.tensor([row[col]], dtype=torch.float) for col in numeric_columns}
        
#         # Zero the gradients, forward pass, compute loss, backward pass, update weights
#         optimizer.zero_grad()
#         log_probs = model({**context_idxs, **numeric_values})
#         target = torch.tensor([row['target']], dtype=torch.float).resize_((1, 1))
#         loss = loss_function(log_probs, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     losses.append(total_loss / len(df))
#     model.eval()
#     with torch.no_grad():
#         total_val_loss = 0
#         for i, row in df_dev.iterrows():
#             context_idxs = {col: torch.tensor([word_to_ix[col][row[col]]], dtype=torch.long) for col in categorical_columns}
#             numeric_values = {col: torch.tensor([row[col]], dtype=torch.float) for col in numeric_columns}
#             log_probs = model({**context_idxs, **numeric_values})
#             val_target = torch.tensor([row['target']], dtype=torch.float).resize_((1, 1))
#             val_loss = loss_function(log_probs, val_target)
#             total_val_loss += val_loss.item()
#         val_losses.append(total_val_loss / len(df_dev))
#     print('Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, losses[-1], val_losses[-1]))

# # # Train the model
# # loss_function = nn.BCELoss()
# # model = CategoricalEmbedder(embeddings, numeric_embeddings)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# # # Assuming you have prepared your data for training
# # # and have encoded the categorical columns into indices

# # # Now iterate over the data with a sliding window of size CONTEXT_SIZE
# # for i in range(CONTEXT_SIZE, len(df) - CONTEXT_SIZE):
# #     # Extract the context words and numeric values for each sample
# #     context_idxs = {col: torch.tensor(word_to_ix[col][df.iloc[i][col]], dtype=torch.long) for col in categorical_columns}
# #     numeric_values = {col: torch.tensor(df.iloc[i][col], dtype=torch.float) for col in numeric_columns}
    
# #     # Zero the gradients, forward pass, compute loss, backward pass, update weights
# #     optimizer.zero_grad()
# #     log_probs = model({**context_idxs, **numeric_values})
# #     loss = loss_function(log_probs, torch.tensor([df.iloc[i]['target']], dtype=torch.float).resize_((1, 1)))
# #     loss.backward()
# #     optimizer.step()
#     # Your training logic here


