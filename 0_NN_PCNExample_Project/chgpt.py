# batching 
# no context size
# bad scores but better than when we dont batch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your data
file_path = "CleanDF.csv"
df = pd.read_csv(file_path)

# Preprocess and encode the target variable
label_encoder = LabelEncoder()
df['procedureAnalysedMedia'] = label_encoder.fit_transform(df['procedureAnalysedMedia'])

# Define your categorical and numeric columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
target_column = 'procedureAnalysedMedia'

# Define mappings for categorical columns
categorical_mappings = {col: {v: k for k, v in enumerate(df[col].unique())} for col in categorical_columns}

# Split the dataset
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)

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
