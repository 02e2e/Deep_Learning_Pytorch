import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
CONTEXT_SIZE = 2  # Number of words to consider as context
EMBEDDING_DIM = 10

# Sample data
data = {
    'multiclass_category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Title': ['This is a sample title', 'Another title for example', 'Yet another lengthy title',
              'A short title', 'Sample title number five', 'This is the sixth title',
              'The seventh title', 'An eighth title sentence', 'Ninth example title text',
              'Final title in the dataset'],
    'quantity': [10, 25, 15, 8, 22, 31, 14, 19, 27, 12],
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing
categories, vocab = set(), set()
for category in df['multiclass_category']:
    categories.add(category)
for title in df['Title']:
    words = title.split()[:CONTEXT_SIZE]
    for word in words:
        vocab.add(word)

category_to_ix = {category: i for i, category in enumerate(categories)}
word_to_ix = {word: i for i, word in enumerate(vocab)}

# Pad and truncate titles
MAX_TITLE_LEN = CONTEXT_SIZE

def pad_title(title, max_len):
    words = title.split()[:max_len] + ['<pad>'] * (max_len - len(words[:max_len]))
    return [word_to_ix[word] for word in words]

# Model
class TitleClassifier(nn.Module):
    def __init__(self, category_size, vocab_size, embedding_dim, context_size):
        super(TitleClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings2 = nn.Embedding(category_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * (context_size + 1) + 1, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        category, title, quantity = inputs
        title_embed = self.embeddings(title).view((1, -1))
        category_embed = self.embeddings2(category).view((1, -1))
        embeds_full = torch.cat((category_embed, title_embed, quantity.unsqueeze(1)), -1)
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs

# Training
losses = []
val_losses = []
loss_function = nn.BCELoss()
model = TitleClassifier(len(categories), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters())

# Split data into train and validation
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

for epoch in range(100):
    total_loss = 0
    for _, row in train_df.iterrows():
        category_idx = torch.tensor([category_to_ix[row['multiclass_category']]], dtype=torch.long)
        padded_title = pad_title(row['Title'], MAX_TITLE_LEN)
        padded_title = torch.tensor(padded_title, dtype=torch.long)
        quantity = torch.tensor([row['quantity']], dtype=torch.float)
        model.zero_grad()
        log_probs = model((category_idx, padded_title, quantity))
        loss = loss_function(log_probs, torch.tensor([row['target']], dtype=torch.float).resize_((1, 1)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for _, row in val_df.iterrows():
            category_idx = torch.tensor([category_to_ix[row['multiclass_category']]], dtype=torch.long)
            padded_title = pad_title(row['Title'], MAX_TITLE_LEN)
            padded_title = torch.tensor(padded_title, dtype=torch.long)
            quantity = torch.tensor([row['quantity']], dtype=torch.float)
            log_probs = model((category_idx, padded_title, quantity))
            loss = loss_function(log_probs, torch.tensor([row['target']], dtype=torch.float).resize_((1, 1)))
            val_loss += loss.item()
        val_losses.append(val_loss)
    print(f'Epoch: {epoch + 1}, Train Loss: {losses[-1]}, Validation Loss: {val_losses[-1]}')

# Print losses
for train_loss, val_loss in zip(losses, val_losses):
    print(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')