# not doing batch sizing or context size in this version
# not very good scores 

# Load Dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "CleanDF.csv"
df = pd.read_csv(file_path)
df = df.drop(columns='Unnamed: 0')

# Data Preparation
df.dtypes
numeric_columns = df.select_dtypes('number').columns
categorical_columns = df.select_dtypes('object').columns
df[df.select_dtypes('object').columns].nunique().reset_index(name='Cardinality')

target_encoder = LabelEncoder()
df['procedureAnalysedMedia'] = target_encoder.fit_transform(df['procedureAnalysedMedia'])
target = 'procedureAnalysedMedia'

categorical_mappings = {col: {v: k for k, v in enumerate(df[col].unique())} for col in categorical_columns}

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(df.drop(columns=[target]), df[target], test_size=0.2, random_state=42)
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
        return categorical, numerical, target.view(-1)

EMBEDDING_DIM = 8
vocab_sizes = [len(df[col].unique()) for col in categorical_columns]

class NNGRAMLanguageModeler(nn.Module):
    def __init__(self, vocab_sizes, num_numeric_cols, embedding_dim):
        super(NNGRAMLanguageModeler, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes])
        self.linear1 = nn.Linear(embedding_dim * len(categorical_columns) + num_numeric_cols, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, categorical_inputs, numerical_inputs):
        categorical_embeddings = [self.embeddings[i](categorical_inputs[:, i]) for i in range(len(categorical_columns))]
        categorical_embeddings = torch.cat(categorical_embeddings, dim=1)
        combined_inputs = torch.cat((categorical_embeddings, numerical_inputs), dim=1)
        x = F.relu(self.linear1(combined_inputs))
        x = torch.sigmoid(self.linear2(x))
        return x

# Training
model = NNGRAMLanguageModeler(vocab_sizes, len(numeric_columns), EMBEDDING_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_dataset = CustomDataset(train_df, categorical_columns, numeric_columns, target, categorical_mappings)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(train_dataset)):
        categorical, numerical, target = train_dataset[i]
        categorical = categorical.unsqueeze(0)  # Add batch dimension
        numerical = numerical.unsqueeze(0)  # Add batch dimension
        target = target.unsqueeze(0)  # Add batch dimension
        
        optimizer.zero_grad()
        output = model(categorical, numerical)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss}')
