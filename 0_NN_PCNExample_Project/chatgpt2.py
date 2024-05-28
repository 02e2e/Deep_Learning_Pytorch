# batching 
# no context size
# bad scores but better than when we dont batch
# different way to do it 



# Load Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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


EMBEDDING_DIM = 8

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

# split the dataset
# Split the dataset
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)

class CustomDataset(Dataset):
    def __init__(self, dataframe, categorical_cols, numeric_cols, target_col, categorical_mappings):
        self.dataframe = dataframe
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_col = target_col
        self.categorical_mappings = categorical_mappings

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        categorical = torch.tensor([self.categorical_mappings[col][row[col]] for col in self.categorical_cols], dtype=torch.long)
        numerical = torch.tensor(row[self.numeric_cols].astype(np.float32).values, dtype=torch.float)
        target = torch.tensor(row[self.target_col], dtype=torch.float)
        # Adjusted here
        return categorical, numerical, target.view(-1, 1)



class MixedInputModel(nn.Module):
    def __init__(self, categorical_vocab_sizes, num_numeric_cols, embedding_dim, output_dim=1):
        super(MixedInputModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in categorical_vocab_sizes])
        self.linear1 = nn.Linear(len(categorical_vocab_sizes) * embedding_dim + num_numeric_cols, 128)
        self.linear2 = nn.Linear(128, output_dim)

    def forward(self, categorical_inputs, numerical_inputs):
        embeddings = [self.embeddings[i](categorical_inputs[:, i]) for i in range(len(self.embeddings))]
        categorical_embeds = torch.cat(embeddings, dim=1)
        combined_inputs = torch.cat((categorical_embeds, numerical_inputs), dim=1)
        x = F.relu(self.linear1(combined_inputs))
        x = torch.sigmoid(self.linear2(x))
        return x

categorical_vocab_sizes = [len(v) for v in categorical_mappings.values()]
model = MixedInputModel(categorical_vocab_sizes, len(numeric_columns), EMBEDDING_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


train_dataset = CustomDataset(train_df, categorical_columns, numeric_columns, target_column, categorical_mappings)
val_dataset = CustomDataset(val_df, categorical_columns, numeric_columns, target_column, categorical_mappings)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for epoch in range(100):  # Adjust the number of epochs
    model.train()
    total_loss = 0
    for categorical, numerical, target in train_loader:
        model.zero_grad()
        output = model(categorical, numerical)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Similar validation loop, but with model.eval() and torch.no_grad()
