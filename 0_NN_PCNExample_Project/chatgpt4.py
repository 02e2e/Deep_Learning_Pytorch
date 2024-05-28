# this one has an error 
#
TypeError: __init__() got an unexpected keyword argument 'EMBEDDING_DIM'





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Assuming df is already loaded and preprocessed as per your code

# Define your categorical and numerical column processing
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
target_column = 'procedureAnalysedMedia'

# Encode your target
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Dictionary to store categorical data mappings
categorical_mappings = {col: {val: i for i, val in enumerate(df[col].unique())} for col in categorical_columns}

# Convert DataFrame to PyTorch Dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        categorical = torch.tensor([categorical_mappings[col].get(row[col], 0) for col in categorical_columns], dtype=torch.long)
        numerical = torch.tensor(row[numeric_columns].values.astype(np.float32), dtype=torch.float)
        target = torch.tensor(row[self.target_column], dtype=torch.float)
        return categorical, numerical, target

# Define the model
class NGramLanguageModeler(nn.Module):
    def __init__(self, categorical_vocab_sizes, num_numeric_cols, embedding_dim):
        super(NGramLanguageModeler, self).__init__()
        self.categorical_embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in categorical_vocab_sizes])
        self.linear1 = nn.Linear(len(categorical_vocab_sizes) * embedding_dim + len(num_numeric_cols), 128)
        self.linear2 = nn.Linear(128, 1)
    
    def forward(self, categorical_data, numerical_data):
        cat_embeds = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.categorical_embeddings)]
        cat_embeds = torch.cat(cat_embeds, 1)
        combined_data = torch.cat((cat_embeds, numerical_data), 1)
        x = F.relu(self.linear1(combined_data))
        x = torch.sigmoid(self.linear2(x))
        return x

# Splitting the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = DataFrameDataset(train_df, target_column)
test_dataset = DataFrameDataset(test_df, target_column)

# Model, Loss, and Optimizer
vocab_sizes = [len(categorical_mappings[col]) for col in categorical_columns]
model = NGramLanguageModeler(vocab_sizes, numeric_columns, EMBEDDING_DIM=10)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    total_loss = 0
    for categorical, numerical, target in train_dataset:
        categorical = categorical.unsqueeze(0)  # Add batch dimension
        numerical = numerical.unsqueeze(0)  # Add batch dimension
        target = target.unsqueeze(0)  # Add batch dimension

        model.zero_grad()
        log_probs = model(categorical, numerical)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch}: Loss {total_loss / len(train_dataset)}')
