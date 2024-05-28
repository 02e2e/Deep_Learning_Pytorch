
# Load Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# file path
file_path = "CleanDF.csv" # previously cleaned
# file_path = '../1 - Visualization and Data Preprocessing/Data/OnlineNewsPopularity.csv' # unclean

# Load the dataset
df = pd.read_csv(file_path)

# Set the maximum number of columns to display to None
pd.set_option('display.max_columns', None)
df.head()
df = df.drop(columns='Unnamed: 0')




######################## Prep data ########################
# look at data types 
df.dtypes
# identify numeric columns 
numeric_columns = df.select_dtypes('number').columns
# identify categorical columns 
categorical_columns = df.select_dtypes('object').columns
# look at the cardinality of the categorical variables
df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')



df['procedureAnalysedMedia'].head() # this will be the target variable 
# label encode the 2 categories (water (1) sediment (0))
target_encoder = LabelEncoder()
df['procedureAnalysedMedia'] = target_encoder.fit_transform(df['procedureAnalysedMedia'])
target = df['procedureAnalysedMedia'] # target variable 
df['procedureAnalysedMedia'].head() # water selected as 1 water, 0 sediment
# maybe we dont drop the label -- we didnt in the example by Santerre 
# df.drop('procedureAnalysedMedia', axis=1, inplace=True) # drop our target variable from the df 
# df.columns() # check 

##########################

 
# categorical columns identified and put into a list 
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# Define the embedding layer for the numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
target_column = 'procedureAnalysedMedia'


# # vocab for categorical variables 
# vocabularies = {col: set(df[col].unique()) for col in categorical_columns}

# you need a word and speaker to ix for each categrocial and numerical column 
# dont forget the target varaibale 
# Define mappings for each categorical column
##############################################
# categorical_mappings = {}
# for col in categorical_columns:
#     categorical_mappings[col] = {category: i for i, category in enumerate(vocabularies[col])}
categorical_mappings = {col: {v: k for k, v in enumerate(df[col].unique())} for col in categorical_columns}
#check the mapping - looks good 
category_index_cat_col_1 = categorical_mappings['parameterWaterBodyCategory']['GW'] # returns 0 
category_index_cat_col_1 = categorical_mappings['parameterWaterBodyCategory']# {'GW': 0, 'LW': 1, 'RW': 2}
category_index_cat_col_2 = categorical_mappings['observedPropertyDeterminandCode'] # 0 to 181 
# Define vocab sizes for each categorical variable
# vocab_sizes = [len(set(df[col])) for col in categorical_columns]

# # split the dataset
# # Split the dataset
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)

# batch the data using torch dataloader


class CustomDataset(Dataset):
    def __init__(self, dataframe, categorical_cols, numeric_cols, target_col):
        self.dataframe = dataframe
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        categorical = torch.tensor([categorical_mappings[col][row[col]] for col in self.categorical_cols], dtype=torch.long)
        numerical = torch.tensor(row[self.numeric_cols].values.astype(np.float32), dtype=torch.float)
        target = torch.tensor(row[self.target_col], dtype=torch.float)
        return torch.cat((categorical, numerical)), target.view(1)



##### from here up is okay i think ### this is where we are at on the actual pcn script #### 






















class NNGRAMLanguageModeler(nn.Module):
    def __init__(self, vocab_sizes, num_numeric_cols, embedding_dim):
        super(NNGRAMLanguageModeler, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes])
        self.linear1 = nn.Linear(embedding_dim * len(categorical_columns) + num_numeric_cols, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        categorical_inputs = inputs[:, :len(categorical_columns)].long()
        numerical_inputs = inputs[:, len(categorical_columns):]
        categorical_embeddings = [self.embeddings[i](categorical_inputs[:, i]) for i in range(len(categorical_columns))]
        categorical_embeddings = torch.cat(categorical_embeddings, dim=1)
        combined_inputs = torch.cat((categorical_embeddings, numerical_inputs), dim=1)
        x = torch.relu(self.linear1(combined_inputs))
        x = torch.sigmoid(self.linear2(x))
        return x

# Hyperparameters
EMBEDDING_DIM = 8
vocab_sizes = [len(set(df[col])) for col in categorical_columns]

# Prepare DataLoaders
train_dataset = CustomDataset(train_df, categorical_columns, numeric_columns, target_column)
val_dataset = CustomDataset(val_df, categorical_columns, numeric_columns, target_column)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss Function, and Optimizer
model = NNGRAMLanguageModeler(vocab_sizes, len(numeric_columns), EMBEDDING_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training and Validation Loops
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for inputs, target in train_dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    print(f'Epoch {epoch+1}, Training Loss: {total_train_loss}')
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            output = model(inputs)
            loss = loss_function(output, target)
            total_val_loss += loss.item()
    print(f'Epoch {epoch+1}, Validation Loss: {total_val_loss}')

####################################################
####################################################
####################################################

##############Tuesday 
# incoporate your train test split 

# incoporate batching 
# run NNGRam 
# run neural network 




class NNGRAMLanguageModeler(nn.Module):
    def __init__(self, vocab_sizes, num_numeric_cols, embedding_dim, context_size):
        super(NNGRAMLanguageModeler, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes])
        self.linear1 = nn.Linear(embedding_dim * len(categorical_columns) + num_numeric_cols, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        # Split inputs back into categorical and numerical inputs
        categorical_inputs = inputs[:, :len(categorical_columns)].long()
        numerical_inputs = inputs[:, len(categorical_columns):]
        categorical_embeddings = [self.embeddings[i](categorical_inputs[:, i]) for i in range(len(categorical_columns))]
        categorical_embeddings = torch.cat(categorical_embeddings, dim=1)
        combined_inputs = torch.cat((categorical_embeddings, numerical_inputs), dim=1)
        x = torch.relu(self.linear1(combined_inputs))
        x = torch.sigmoid(self.linear2(x))
        return x


EMBEDDING_DIM = 8
CONTEXT_SIZE = 3  # Adjust based on your model's specific design
vocab_sizes = [len(df[col].unique()) for col in categorical_columns]

# Dataset and DataLoader
dataset = CustomDataset(df, categorical_columns, numeric_columns, target_column)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, and Optimizer
model = NNGRAMLanguageModeler(vocab_sizes, len(numeric_columns), EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, target in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss}')




#########################
# Define vocab sizes for each categorical variable
vocab_sizes = [len(set(df[col])) for col in categorical_columns]

# word embed the cat variables
# numeric variables will be normalized 
# seperate the target variable from the input features




############ from here up is perfect ############ Tuesday 3pm 




######## working here down ########


# Define NNGRAM-based neural network model
class NNGRAMLanguageModeler(nn.Module):
    def __init__(self, vocab_sizes, num_numeric_cols, embedding_dim, context_size):
        super(NNGRAMLanguageModeler, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes])
        self.linear1 = nn.Linear(embedding_dim * (len(categorical_columns) + context_size) + num_numeric_cols, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        categorical_inputs = [self.embeddings[i](inputs[:, i].long()) for i in range(len(self.embeddings))]  # Embed each categorical variable
        categorical_inputs = torch.cat(categorical_inputs, dim=1)  # Concatenate categorical embeddings
        numerical_inputs = inputs[:, len(categorical_columns):]  # Select numerical inputs
        embeds = torch.cat((categorical_inputs, numerical_inputs), dim=1)  # Concatenate categorical and numerical inputs
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs

# Define your model
CONTEXT_SIZE = 3  # Adjust as needed
EMBEDDING_DIM = 8  # Adjust as needed

# model = NNGRAMLanguageModeler(vocab_sizes, len(numeric_columns), EMBEDDING_DIM, CONTEXT_SIZE)

# Print the model architecture
print(model)

####
# Define the loss function and instantiate the model
loss_function = nn.BCELoss()
model = NNGRAMLanguageModeler(vocab_sizes, len(numeric_columns), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Loop through epochs
num_epochs = 10  # Adjust as needed
losses = []
for epoch in range(num_epochs):
    total_loss = 0
    # Loop through your data
    for index, row in df.iterrows():
        # Extract inputs and target from the current row
        categorical_inputs = [categorical_mappings[col][row[col]] for col in categorical_columns]
        numerical_inputs = row[numeric_columns].values
        target = row['procedureAnalyzedMedia']  # Assuming this is your target column
        inputs = np.concatenate([categorical_inputs, numerical_inputs])
        
        # Convert inputs and target to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float)
        target_tensor = torch.tensor([target], dtype=torch.float)
        
        # Zero the gradients, forward pass, compute loss, backward pass, and update parameters
        optimizer.zero_grad()
        log_probs = model(inputs_tensor)
        loss = loss_function(log_probs, target_tensor)
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
    
    # Append total loss for the epoch to the losses list
    losses.append(total_loss)
    
    # Print the total loss for the epoch
    print(f"Epoch {epoch + 1}, Total Loss: {total_loss}")

# Print the list of losses for each epoch
print(losses)












######## working here UP ########
















###########here on MOnday 3/25





# And so on for cat_col3, cat_col4, cat_col5, cat_col6



###########
###########
############

#
# Define the embedding layer for the numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


word_to_ix = {}
emebeddings = {}
# create a a word index to mapping for each categorical column and embeddings 
for col in categorical_columns: 
    word_to_ix[col] = {word: i for i, word in enumerate(vocabularies[col])}
    # define the embedding layers for each categorical column 
    embeddings[dim] = nn.Embedding(len(vocabularies[col]), EMEBEDDING_DIM)


    



##########################################



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