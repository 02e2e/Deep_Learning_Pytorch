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



# Now we are here below 
def collate_fn(batch):
    sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target = zip(*batch)
    
    sentence_data_padded = torch.stack(sentence_data_padded)
    company_data_padded = torch.stack(company_data_padded)
    numeric_data = torch.stack(numeric_data) 
    multi_class_cat_data = torch.stack(multi_class_cat_data)
    # target = torch.stack(target)
    target = torch.stack(target).view(-1, 1)  # Ensure shape [batch_size, 1]    
    
    return sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data, target


# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)

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
print([len(seq) for seq in sentence_data_padded])

print(sentence_data_padded.shape, company_data_padded.shape, numeric_data.shape, multi_class_cat_data.shape)
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size_1, vocab_size_2, num_colors, num_numeric_features, embedding_dim=10):
  
        super(WordEmbeddingModel, self).__init__()
        
        self.embedding_1 = nn.Embedding(vocab_size_1, embedding_dim)
        self.embedding_2 = nn.Embedding(vocab_size_2, embedding_dim)
        self.multi_cat_embedding = nn.Embedding(num_colors, embedding_dim)

        
        input_dim = 2 * embedding_dim + num_colors + num_numeric_features
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 1)
# here we take the data from that the collate_fn function returns
    def forward(self, sentence_data_padded, company_data_padded, numeric_data, multi_class_cat_data):
        # Process the variable-length sequences
        print(f"sentence_data_padded dtype: {sentence_data_padded.dtype}")
        print(f"company_data_padded dtype: {company_data_padded.dtype}")
        print(f"numeric_data dtype: {numeric_data.dtype}")
        print(f"multi_class_cat_data dtype: {multi_class_cat_data.dtype}")
        # lengths = [len(seq) for seq in sentence_data_padded]
        # text_1_packed = rnn_utils.pack_padded_sequence(self.embedding_1(sentence_data_padded), lengths= lengths, batch_first=True, enforce_sorted=False)
        # company_lengths = [len(seq) for seq in company_data_padded]
        # text_2_packed = rnn_utils.pack_padded_sequence(self.embedding_2(company_data_padded), company_lengths, batch_first=True, enforce_sorted=False)
        text_1_packed = self.embedding_1(sentence_data_padded)
        text_2_packed = self.embedding_1(sentence_data_padded)
        print(f"text_1_packed shape after packing: {text_1_packed.data.shape}")
        print(f"text_2_packed shape after packing: {text_1_packed.data.shape}")

        # Process the packed sequences
        embed_1, _ = rnn_utils.pad_packed_sequence(text_1_packed, batch_first=True)
        # embed_1 = embed_1.mean(dim=1)  # Averaging over the sequence dimension

        embed_2, _ = rnn_utils.pad_packed_sequence(text_2_packed, batch_first=True)
        # embed_2 = embed_2.mean(dim=1)  # Averaging over the sequence dimension
        print(f"embed_1 shape after padding: {embed_1.shape}")
        print(f"embed_2 shape after padding: {embed_2.shape}")

        numeric_ = numeric_data.unsqueeze(1).repeat(1, self.embedding_dim)  # Assuming numeric_data is a single number
        embed_multi_cat = self.multi_cat_embedding(multi_class_cat_data).mean(dim=1).unsqueeze(1)
# Take the mean after unsqueezing
        
        combined_embeds = torch.cat((embed_1, embed_2, embed_multi_cat, numeric_), dim=1)
        
        out = F.relu(self.linear1(combined_embeds))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs



model = WordEmbeddingModel(vocab_size_1, vocab_size_2, num_colors, num_numeric_features, embedding_dim=20)
print(model)
loss_function = nn.BCEWithLogitsLoss()  # Binary classification

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust optimizer if needed

##########################################################
# Training and Evaluation

# kf = KFold(n_splits=5)
X = list(range(len(dataset)))  # Indices for the dataset

from sklearn.model_selection import KFold

kf = KFold(n_splits=3)  # Ensure n_splits is not greater than the number of samples
X = list(range(len(dataset)))  # Indices for the dataset
y = target_data  # Your target data
cv_accuracies = []

for train_index, test_index in kf.split(X):
    # Split data
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Training
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

# Calculate and print average accuracy
avg_accuracy = sum(cv_accuracies) / len(cv_accuracies)
print(f"Average Cross-Validation Accuracy: {avg_accuracy}")

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





    # Training
#     model.train()
#     for epoch in range(10):  # Lower number of epochs for brevity
#         total_loss = 0
#         for text_1, text_2, numeric_features, multi_class_cat_data, target in train_loader:
#             # Ensure correct shape of target
#             target = target.unsqueeze(1).float()
            
#             # Training Step
#             optimizer.zero_grad()
#             log_probs = model(text_1, text_2, numeric_features, multi_class_cat_data)
#             loss = loss_function(log_probs, target)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss}")
    
#     # Evaluation
#     model.eval()
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for text_1, text_2, numeric_features, multi_class_cat_data, target in test_loader:
#             # Ensure correct shape of target
#             target = target.unsqueeze(1).float()
            
#             log_probs = model(text_1, text_2, numeric_features, multi_class_cat_data)            
#             preds = (log_probs > 0.5).float()
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(target.cpu().numpy())
        
#     accuracy = accuracy_score(all_targets, all_preds)
#     cv_accuracies.append(accuracy)
#     print(f"Fold Accuracy: {accuracy}")

# print(f"Mean CV Accuracy: {np.mean(cv_accuracies)}")








# ##########################################################
# # K-fold cross-validation
# # kf = KFold(n_splits=5)
# X = list(range(len(dataset)))  # Indices for the dataset

# from sklearn.model_selection import KFold

# kf = KFold(n_splits=3)  # Ensure n_splits is not greater than the number of samples
# X = list(range(len(dataset)))  # Indices for the dataset
# y = target_data  # Your target data
# cv_accuracies = []
# for train_index, test_index in kf.split(X):
#     # Split data
#     train_data = torch.utils.data.Subset(dataset, train_index)
#     test_data = torch.utils.data.Subset(dataset, test_index)
    
#     train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
#     # Training
#     model.train()
#     for train_index, test_index in kf.split(X):
#         # Your existing code for training and evaluation goes here

#         # Split data
#         train_data = torch.utils.data.Subset(dataset, train_index)
#         test_data = torch.utils.data.Subset(dataset, test_index)
        
#         train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
#         test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
#         # Training
#         model.train()

#         for epoch in range(10):  # Lower number of epochs for brevity
#             total_loss = 0
#             for text_1, text_2, numeric_features, multi_class_cat_data, target in train_loader:
#                 # Ensure correct shape of target
#                 target = target.unsqueeze(1).float()
            
#                 # Training Step
#                 optimizer.zero_grad()
#                 log_probs = model(text_1, text_2, numeric_features, multi_class_cat_data)
#                 loss = loss_function(log_probs, target)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"Epoch {epoch+1}, Loss: {total_loss}")
        
#         # Evaluation
#         model.eval()
#         all_preds = []
#         all_targets = []
#         with torch.no_grad():
#             for text_1, text_2, numeric_features, multi_class_cat_data, target in test_loader:
#                 # Ensure correct shape of target
#                 target = target.unsqueeze(1).float()
                
#                 log_probs = model(text_1, text_2, numeric_features, multi_class_cat_data)            
#                 preds = (log_probs > 0.5).float()
#                 all_preds.extend(preds.cpu().numpy())
#                 all_targets.extend(target.cpu().numpy())
        
#         accuracy = accuracy_score(all_targets, all_preds)
#         cv_accuracies.append(accuracy)
#         print(f"Fold Accuracy: {accuracy}")

#     print(f"Mean CV Accuracy: {np.mean(cv_accuracies)}")

















# # ############


