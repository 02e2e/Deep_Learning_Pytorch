##########################################################
# Neural Network with Word Embeddings and Multiple Inputs
##########################################################
# Load Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
import torch.nn.utils.rnn as rnn_utils
##########################################################
# Create Data 
##########################################################
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

##########################################################
# Label Encoding and Normalization
##########################################################
df.dtypes

# Apply StandardScaler normalization to numeric columns
scaler = StandardScaler()
df['quantity'] = scaler.fit_transform(df['quantity'].values.reshape(-1, 1))

# label encoding our target variable to a 0/1 
le = LabelEncoder()
# encode = OneHotEncoder()
df['color'] = le.fit_transform(df['color'])

# Extract target variable values from the DataFrame
target_data = df['target'].values  # This creates a NumPy array of the target variable

# drop the target variable 
df.drop('target', axis=1, inplace=True) 
numeric_data = df['quantity'].values
multi_class_cat_data = df['color'].values

##########################################################
# Tokenization and Padding
##########################################################
CONTEXT_SIZE = 2  
companies, company_vocab = [], []  
for company in df['company']:
    words = company.lower().split()
    companies.append(words)         # Store list of words for each company
    company_vocab.extend(words)     # Extend the vocabulary list with individual words
    for i in range(max(0, len(words) - CONTEXT_SIZE + 1)): 
        context_window = words[i: i + CONTEXT_SIZE]
        company_vocab.extend(context_window)   # Extend the vocabulary with words from the context window

# word_to_ix = {word: i for i, word in enumerate(set(company_vocab))}  # Remove duplicates for indexing
word_to_ix = {word: i+1 for i, word in enumerate(set(company_vocab))}  # Start indices from 1

MAX_LENGTH = 4  # Set a maximum length for company names
# def sentence_to_indices(words, word_to_ix):
#     indices = [word_to_ix[word] for word in words if word in word_to_ix]
#     indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
#     return indices
# Adjust the sentence_to_indices function
def sentence_to_indices(words, word_to_ix):
    indices = [word_to_ix[word] for word in words if word in word_to_ix]
    indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
    return indices

company_data = [sentence_to_indices(row, word_to_ix) for row in companies]

print(company_data)   # This should now give you the correct indices

company_data_padded = pad_sequence([torch.tensor(seq) for seq in company_data], batch_first=True, padding_value=0)
company_data_padded.shape
# torch.Size([20, 8])

##########################################################

CONTEXT_SIZE = 4
sentences, sentences_vocab = [], []  
for sentence in df['sentence']:
    words = sentence.lower().split()
    sentences.append(words)         # Store list of words for each company
    sentences_vocab.extend(words)     # Extend the vocabulary list with individual words
    for i in range(max(0, len(words) - CONTEXT_SIZE + 1)): 
        context_window = words[i: i + CONTEXT_SIZE]
        sentences_vocab.extend(context_window)   # Extend the vocabulary with words from the context window

# word_to_ix1 = {word: i for i, word in enumerate(set(sentences_vocab))}  # Remove duplicates for indexing
word_to_ix1 = {word: i+1 for i, word in enumerate(set(sentences_vocab))}  # Start indices from 1

MAX_LENGTH = 8 # Set a maximum length for company names
# def sentence_to_indices(words, word_to_ix1):
#     indices = [word_to_ix1[word] for word in words if word in word_to_ix1]
#     indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
#     return indices
def sentence_to_indices(words, word_to_ix1):
    indices = [word_to_ix1[word] for word in words if word in word_to_ix1]
    indices += [0] * (MAX_LENGTH - len(indices))  # Pad with zeros
    return indices

sentence_data = [sentence_to_indices(row, word_to_ix1) for row in sentences]

print(sentence_data) 
sentence_data_padded = pad_sequence([torch.tensor(seq) for seq in sentence_data], batch_first=True, padding_value=0)

sentence_data_padded.shape
# torch.Size([20, 10])


##########################################################
# Tensorize, Pad, and Create DataLoader
##########################################################
class WordEmbedDataset(Dataset):
    def __init__(self, sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target_data):
        self.sentence_data_padded = sentence_data_padded
        self.company_data_padded = company_data_padded
        self.numeric_data = numeric_data
        self.multi_class_cat_data = multi_class_cat_data
        self.target_data = target_data

    def __getitem__(self, index):
        sentence_data_padded = torch.tensor(self.sentence_data_padded[index], dtype=torch.long)
        company_data_padded = torch.tensor(self.company_data_padded[index], dtype=torch.long)
        # numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32).unsqueeze(0)  # Ensure shape [1]
        numeric_data = torch.tensor(self.numeric_data[index], dtype=torch.float32)
        multi_class_cat_data = torch.tensor(self.multi_class_cat_data[index], dtype=torch.long)
        # target = torch.tensor(self.target_data[index], dtype=torch.float32).unsqueeze(0)  # Ensure shape [1]
        target = torch.tensor(self.target_data[index], dtype=torch.float32)
        return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target
    
    def __len__(self):
        return len(self.sentence_data_padded)

# Create dataset
dataset = WordEmbedDataset(sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target_data)
# Create DataLoader


def collate_fn(batch):
    sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = zip(*batch)
    
    sentence_data_padded = torch.stack(sentence_data_padded)
    company_data_padded = torch.stack(company_data_padded)
    numeric_data = torch.stack(numeric_data).unsqueeze(1)  # Add an extra dimension
    multi_class_cat_data = torch.stack(multi_class_cat_data).unsqueeze(1)  # Add an extra dimension
    target = torch.stack(target).view(-1, 1)  # Ensure shape [batch_size, 1]    
    
    return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target


# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=False)
###########
# check the shapes 
# Get the first batch of data
first_batch = next(iter(dataloader))

# Unpack the batch
sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = first_batch

# Print the shapes of the tensors
print("sentence_data_padded shape:", sentence_data_padded.shape)
print("company_data_padded shape:", company_data_padded.shape)
print("numeric_data shape:", numeric_data.shape)
print("multi_class_cat_data shape:", multi_class_cat_data.shape)
print("target shape:", target.shape)

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

##########################################################
# Define Model
##########################################################
vocab_size_1 = len(word_to_ix) 
print(vocab_size_1)
vocab_size_2 = len(word_to_ix1)
print(vocab_size_2)
num_colors = len(set(df['color']))
print(num_colors)
num_numeric_features = 1  #
print(vocab_size_1, vocab_size_2, num_colors, num_numeric_features )
print([len(seq) for seq in sentence_data_padded])
print([len(seq) for seq in company_data_padded])

print(sentence_data_padded.shape, company_data_padded.shape, numeric_data.shape, multi_class_cat_data.shape)

# For sentence_data_padded
max_value_sentence = max([max(seq) for seq in sentence_data_padded])
print(f"Max value in sentence_data_padded: {max_value_sentence}")
print(f"Is max value in sentence_data_padded < vocab_size_1? {max_value_sentence < vocab_size_1}")

# For company_data_padded
max_value_company = max([max(seq) for seq in company_data_padded])
print(f"Max value in company_data_padded: {max_value_company}")
print(f"Is max value in company_data_padded < vocab_size_2? {max_value_company < vocab_size_2}")


vocab_size_2 = max_value_company + 1
vocab_size_2
if torch.is_tensor(vocab_size_2):
    vocab_size_2 = vocab_size_2.item()
print(vocab_size_1, vocab_size_2, num_colors, num_numeric_features )
# Remember that the indices in your padded sequences should always be less than the vocabulary size, since they're used to index into the embedding layer which has a size equal to the vocabulary size.

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size_1, vocab_size_2, num_colors, num_numeric_features, embedding_dim=10):
  
        super(WordEmbeddingModel, self).__init__()
        
        self.embedding_1 = nn.Embedding(vocab_size_1, embedding_dim)
        self.embedding_2 = nn.Embedding(vocab_size_2, embedding_dim)
        self.multi_cat_embedding = nn.Embedding(num_colors, embedding_dim)

        input_dim = 3 * embedding_dim + num_numeric_features
        
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 1)
        
        
        
    def forward(self, sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data):
        # Process the variable-length sequences
        embed_1 = self.embedding_1(sentence_data_padded)
        embed_2 = self.embedding_2(company_data_padded)

        # Process the embeddings as needed
        embed_1 = embed_1.mean(dim=1)  # Assuming sentence_data_padded is padded to [batch_size, max_length]
        embed_2 = embed_2.mean(dim=1)  # Assuming company_data_padded is padded to [batch_size, max_length]

        # Other processing steps
        numeric_features = numeric_data  # Assuming numeric_data is of shape [batch_size, num_numeric_features]
        # embed_multi_cat = self.multi_cat_embedding(multi_class_cat_data).mean(dim=1).unsqueeze(1)  # Take the mean after unsqueezing
        embed_multi_cat = self.multi_cat_embedding(multi_class_cat_data).mean(dim=1)
        # Combine the embeddings and other features
        combined_embeds = torch.cat((embed_1, embed_2, embed_multi_cat, numeric_features), dim=1)

        # Forward pass through the linear layers
        out = F.relu(self.linear1(combined_embeds))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs


model = WordEmbeddingModel(vocab_size_1, vocab_size_2, num_colors, num_numeric_features, embedding_dim=20)
print(model)
loss_function = nn.BCEWithLogitsLoss()  # Binary classification

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust optimizer if needed

from torch.nn import functional as F
##########################################################
# Training and Evaluation
##########################################################
kf = KFold(n_splits=5)
X = list(range(len(dataset)))  # Indices for the dataset

from sklearn.model_selection import KFold

kf = KFold(n_splits=3)  # Ensure n_splits is not greater than the number of samples
X = list(range(len(dataset)))  # Indices for the dataset
y = target_data  # Your target data
cv_accuracies = []

def evaluate_model(test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target in test_loader:
            log_probs = model(sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data)
            # Assuming binary classification: predict class based on threshold (0.5)
            predicted = (torch.sigmoid(log_probs) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy


for train_index, test_index in kf.split(X):
    # Split data
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

#     # Training
    model.train()
    for epoch in range(10):  # Lower number of epochs for brevity
        total_loss = 0
        for sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target in train_loader:
            # Training Step (as described earlier)
            optimizer.zero_grad()
            log_probs = model(sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {total_loss}")

    # Evaluation
    model.eval()
    accuracy = evaluate_model(test_loader)  # Replace with your evaluation function
    cv_accuracies.append(accuracy)

# # Calculate and print average accuracy
avg_accuracy = sum(cv_accuracies) / len(cv_accuracies)
print(f"Average Cross-Validation Accuracy: {avg_accuracy}")


##########################################################