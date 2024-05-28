
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import string

# file path
file_path = "CleanDF.csv" 
# file_path = 'tim_clean_mashable_data_01.csv'# previously cleaned
# file_path = '../1 - Visualization and Data Preprocessing/Data/OnlineNewsPopularity.csv' # unclean

# Load the dataset
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

# label encoding our target variable to a 0/1 
target_encoder = LabelEncoder()
df['procedureAnalysedMedia'] = target_encoder.fit_transform(df['procedureAnalysedMedia'])
# target_column = df['procedureAnalysedMedia']

# Extract target variable values from the DataFrame
target_data = df['procedureAnalysedMedia'].values  # This creates a NumPy array of the target variable

# drop the target variable 
df.drop('procedureAnalysedMedia', axis=1, inplace=True)
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

# Apply StandardScaler normalization to numeric columns
scaler = StandardScaler()
df[numeric_columns_list] = scaler.fit_transform(df[numeric_columns_list])

##########################
#  target procedureAnalysedMedia
# special categoricals that need word embedding observedPropertyDeterminandCode and resultUom
##########################


# turn dataframe into a diciontary and go the path from the template from pytorch 

dict_df = df.to_dict('list')

############Dictionary_version################ Wed. 2:44PM 
# For the multi-word categorical column, tokenize and build a vocabulary similarly to the vocab creation in your template. 
# For other categorical columns, directly map unique values to indices.

# Tokenize multi-word categorical column and build vocabularies observedPropertyDeterminandCode 
vocab1 = set(word for row in dict_df['observedPropertyDeterminandCode'] for word in row.split())
word_to_ix1 = {word: i for i, word in enumerate(vocab1)}
# tokenzie multi-word categoprical 2 resultUom
vocab2 = set(word for row in dict_df['resultUom'] for word in row.split())
word_to_ix2 = {word: i for i, word in enumerate(vocab2)}

# For all other categorical columns, build a simple mapping
simple_cat_columns_list = list(df[['parameterWaterBodyCategory','procedureAnalysedFraction', 'parameterSamplingPeriod', 'waterBodyIdentifier', 'Country']])

simple_cat_columns = simple_cat_columns_list
complex_cat_columns = longer_cat_col_list
categorical_columns = categorical_columns_list
numeric_columns = numeric_columns_list
# categorical_columns = ['cat1', 'cat2', 'cat3', 'cat4', 'multi_word_col']  # Example categorical columns

# Example categorical columns
 # Example categorical columns with mutli-word columns as well that were word embeddded observedPropertyDeterminandCode and resultUom

# YOU ARE HERE - START HERE THURSDAY #######################

############### mappings #################Thursday
# For single-word categorical columns, build a simple mapping directly from the DataFrame dictionary
simple_mappings = {col: {v: k for k, v in enumerate(set(dict_df[col]))} for col in categorical_columns if col not in ['observedPropertyDeterminandCode', 'resultUom']}
# Combine the simple mappings with the pre-built vocabularies for multi-word categorical columns
mappings = {**simple_mappings, 'observedPropertyDeterminandCode': word_to_ix1, 'resultUom': word_to_ix2}
# target_column = 'procedureAnalysedMedia'  # Assuming the target variable is named 'target_column'
# mappings = {col: {v: k for k, v in enumerate(set(dict_df[col]))} for col in categorical_columns if col != 'multi_word_col'}

# word embedding - Applying mappings to convert sentences to sequences of indices: For each sentence in your multi-word categorical columns, you convert the sentence into a sequence of indices using your mappings. This process is akin to preparing data for language models where each word in a sentence is represented by its index in the vocabulary.

for col, mapping in simple_mappings.items():
    dict_df[col] = [mapping.get(val, 0) for val in dict_df[col]]  # Using get() to handle unseen values gracefully
dict_df['observedPropertyDeterminandCode'] = [[word_to_ix1.get(word, 0) for word in sentence.split()] for sentence in dict_df['observedPropertyDeterminandCode']]
dict_df['resultUom'] = [[word_to_ix2.get(word, 0) for word in sentence.split()] for sentence in dict_df['resultUom']]

dict_df = dict_df.to_dict(orient='records')

#### from here up is good to go !!! ### data preprocessing is complete 


# new 
# class CustomDataset(Dataset):
#     def __init__(self, dict_df, numeric_columns, simple_cat_columns, target_data):
#         self.dict_df = dict_df  # Use the entire dict for flexibility
#         self.numeric_columns = numeric_columns
#         self.simple_cat_columns = simple_cat_columns
#         self.target_data = target_data

#     def __len__(self):
#         return len(self.target_data)

#     def __getitem__(self, idx):
#         # Gather numeric data
#         numeric_data = {col: self.dict_df[col][idx] for col in self.numeric_columns}
#         numeric_data_tensor = torch.tensor([value for _, value in numeric_data.items()], dtype=torch.float32)
        
#         # Gather simple categorical data
#         simple_cat_data = {col: self.dict_df[col][idx] for col in self.simple_cat_columns}
#         simple_cat_data_tensor = torch.tensor([value for _, value in simple_cat_data.items()], dtype=torch.long)  # Assuming they're already encoded as integers
        
#         # Handle longer categorical variables separately as before
#         observedPropertyDeterminandCode = torch.tensor(self.dict_df['observedPropertyDeterminandCode'][idx], dtype=torch.long)
#         resultUom = torch.tensor(self.dict_df['resultUom'][idx], dtype=torch.long)
        
#         # Return a combined dictionary
#         return {
#             'numeric_data': numeric_data_tensor,
#             'simple_cat_data': simple_cat_data_tensor,
#             'observedPropertyDeterminandCode': observedPropertyDeterminandCode,
#             'resultUom': resultUom,
#             'target': torch.tensor(self.target_data[idx], dtype=torch.float32)
#         }

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data):
        self.dict_df = dict_df
        self.numeric_columns = numeric_columns
        self.simple_cat_columns = simple_cat_columns
        self.complex_cat_columns = complex_cat_columns  # complex_cat_columns is a dict with keys as column names and values as mappings
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        # Handling numeric data
        numeric_data = [self.dict_df[col][idx] for col in self.numeric_columns]
        numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32)

        # Handling simple categorical data (already encoded)
        simple_cat_data = [self.dict_df[col][idx] for col in self.simple_cat_columns]
        simple_cat_tensor = torch.tensor(simple_cat_data, dtype=torch.long)

        # Handling complex categorical data (requiring embeddings)
        complex_cat_tensors = []
        for col, mapping in self.complex_cat_columns.items():
            encoded_value = mapping[self.dict_df[col][idx]]  # Assume dict_df[col][idx] is directly mappable
            complex_cat_tensors.append(torch.tensor(encoded_value, dtype=torch.long))

        target = torch.tensor(self.target_data[idx], dtype=torch.float32)

        return {
            'numeric_data': numeric_tensor,
            'simple_cat_data': simple_cat_tensor,
            'complex_cat_data': torch.stack(complex_cat_tensors),  # Stack complex categorical tensors
            'target': target
        }
        

# new 
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    numeric_data = torch.stack([item['numeric_data'] for item in batch])
    simple_cat_data = torch.stack([item['simple_cat_data'] for item in batch])
    complex_cat_data = [item['complex_cat_data'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])

    # Padding the complex categorical data
    # Assuming complex_cat_data is a list of tensors where each tensor can have a different length
    padded_complex_cat_data = pad_sequence(complex_cat_data, batch_first=True, padding_value=0)
    
    return {
        'numeric_data': numeric_data,
        'simple_cat_data': simple_cat_data,
        'complex_cat_data': padded_complex_cat_data,
        'target': targets
    }

dataset = CustomDataset(dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


# train / test split 
from sklearn.model_selection import train_test_split

# Assuming your data is already loaded into variables: dict_df, numeric_columns, simple_cat_columns, complex_cat_columns, target_data

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dict_df, target_data, test_size=0.2, random_state=42)

# Creating datasets for training and testing
train_dataset = CustomDataset(X_train, numeric_columns, simple_cat_columns, complex_cat_columns, y_train)
test_dataset = CustomDataset(X_test, numeric_columns, simple_cat_columns, complex_cat_columns, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)



losses = []
val_losses = []
loss_function = nn.BCELoss()
model = NGramLanguageModeler(len(speakers), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)










###################
###################
###################
# original below 

class CustomDataset(Dataset):
    def __init__(self, dict_df, numeric_columns, target_data):
        # Validation check: Ensure all columns have the same length
        sample_length = len(dict_df[numeric_columns[0]])
        for col in numeric_columns + ['observedPropertyDeterminandCode', 'resultUom']:
            if len(dict_df[col]) != sample_length:
                raise ValueError(f"Column '{col}' length does not match other columns.")
        
        # Assuming numeric data can be safely converted to a NumPy array
        self.numeric_data = np.array([dict_df[col] for col in numeric_columns], dtype=np.float32).T
        # Assuming target_data is passed as a list or numpy array
        self.target_data = np.array(target_data, dtype=np.float32)
        # Store tokenized sequences as lists of integers for dynamic padding
        self.observedPropertyDeterminandCode = dict_df['observedPropertyDeterminandCode']
        self.resultUom = dict_df['resultUom']
        
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        # Return numeric data, tokenized sequences as lists, and target for each item
        return {
            'numeric_data': torch.tensor(self.numeric_data[idx], dtype=torch.float),
            'observedPropertyDeterminandCode': self.observedPropertyDeterminandCode[idx],
            'resultUom': self.resultUom[idx],
            'target': torch.tensor(self.target_data[idx], dtype=torch.float)
        }

from torch.nn.utils.rnn import pad_sequence

# pad for those those categorical variables that require padding 
def collate_fn(batch):
    numeric_data = torch.stack([item['numeric_data'] for item in batch])
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float)

    observedPropertyDeterminandCode = pad_sequence([torch.tensor(item, dtype=torch.long) for item in [b['observedPropertyDeterminandCode'] for b in batch]], batch_first=True, padding_value=0)
    resultUom = pad_sequence([torch.tensor(item, dtype=torch.long) for item in [b['resultUom'] for b in batch]], batch_first=True, padding_value=0)

    # Return a single batch object
    return {
        'numeric_data': numeric_data,
        'observedPropertyDeterminandCode': observedPropertyDeterminandCode,
        'resultUom': resultUom,
        'targets': targets
    }



# Assuming target_data is your target variable list or numpy array
custom_dataset = CustomDataset(dict_df, numeric_columns, target_data)

# DataLoader remains unchanged
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# dataloader = DataLoader(CustomDataset(dict_df, numeric_columns, target_column), batch_size=32, shuffle=True, collate_fn=collate_fn)


############### here TUESDAY good from here up !!! #############
# Start here Wednesday 
# model suggestion - 4/2
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
class MyModel(nn.Module):
    def __init__(self, num_numeric_features, num_categories_list):
        super(MyModel, self).__init__()
        self.embedding_dim = 10  # Set the embedding dimension
        
        # Initialize embeddings with embedding_dim for each categorical variable
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, self.embedding_dim) for num_categories in num_categories_list])
        
        # Calculate the total size of concatenated features: numeric features + all embeddings
        total_feature_size = num_numeric_features + len(num_categories_list) * self.embedding_dim
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(total_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )



    def forward(self, x_numeric, x_categorical):
            # Embedding categorical variables and reshaping
            cat_embeddings = []
            for i, cat_var in enumerate(x_categorical):
                embedded = self.embeddings[i](cat_var)
                cat_embeddings.append(embedded)
            
            # Flatten the embeddings and concatenate with numeric features
            cat_embeddings = torch.cat(cat_embeddings, dim=-1)
            combined = torch.cat([x_numeric, cat_embeddings], dim=1)
            
            # Pass through fully connected layers
            return self.fc_layers(combined)
    
    
    # def forward(self, x_numeric, x_categorical):
    #     # Embed categorical variables and concatenate with numeric features
    #     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
    #     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
    #     return self.fc_layers(x)


# def forward(self, x_numeric, x_categorical):
#     # Diagnostics: Check max index for each categorical variable
#     for i, cat_var in enumerate(x_categorical):
#         max_index = torch.max(cat_var)
#         if max_index >= self.embeddings[i].num_embeddings:
#             raise ValueError(f"Out of range index {max_index} for embedding layer {i} with size {self.embeddings[i].num_embeddings}")
    
#     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
#     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
#     return self.fc_layers(x)

# def forward(self, x_numeric, x_categorical):
#     for i, (embedding, cat_var) in enumerate(zip(self.embeddings, x_categorical)):
#         max_index = torch.max(cat_var)
#         print(f"Max index for embedding {i}: {max_index.item()} (Size: {embedding.num_embeddings})")
#         assert max_index < embedding.num_embeddings, f"Index out of range for embedding {i}"

#     cat_embeddings = [embedding(cat_var).view(cat_var.size(0), -1) for embedding, cat_var in zip(self.embeddings, x_categorical)]
#     x = torch.cat([x_numeric] + cat_embeddings, dim=1)
#     return self.fc_layers(x)


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)  # Using 0.5 as threshold
    return (y_true == predicted).sum().float() / len(y_true)



# Train/Test Split 
from sklearn.model_selection import train_test_split

# Convert dict_df back to DataFrame for easy manipulation
df = pd.DataFrame(dict_df)
# Split data
X_train, X_test, y_train, y_test = train_test_split(df, target_data, test_size=0.2, random_state=42)

# Convert splits back to the required format if necessary
train_dict_df = X_train.to_dict('list')
test_dict_df = X_test.to_dict('list')

# Initialize custom datasets
train_dataset = CustomDataset(train_dict_df, numeric_columns, y_train)
test_dataset = CustomDataset(test_dict_df, numeric_columns, y_test)

# Initialize DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)



# Initialize MyModel with the appropriate number of numeric features and a list of the number of unique categories per categorical variable
model = MyModel(num_numeric_features=3, num_categories_list=[10, 20])  # Adjust these values based on your actual data
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Modify the training loop
for epoch in range(100):
    model.train()
    total_loss, total_accuracy = 0, 0
    
    for batch in train_dataloader:
        # Assuming this unpacking matches your data structure
        x_numeric, x_categorical, targets = batch['numeric_data'], [batch['observedPropertyDeterminandCode'], batch['resultUom']], batch['targets']
        
        optimizer.zero_grad()
        predictions = model(x_numeric, x_categorical)
        loss = loss_function(predictions.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(targets, predictions.squeeze()).item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = total_accuracy / len(train_dataloader)
    print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}')

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_accuracy = sum(calculate_accuracy(batch['targets'], model(batch['numeric_data'], [batch['cat_var1'], batch['cat_var2']]).squeeze()).item() for batch in test_dataloader) / len(test_dataloader)
    print(f'Test Accuracy: {test_accuracy:.4f}')

###############










# Assuming `data_loader` is your DataLoader instance providing batches in the correct format
# for epoch in range(100):
#     total_loss = 0
#     for batch in data_loader:
#         x_numeric, x_categorical, targets = batch['numeric_data'], [batch['cat_var1'], batch['cat_var2']], batch['targets']  # Adapt as necessary

#         model.zero_grad()
#         predictions = model(x_numeric, x_categorical)
#         loss = loss_function(predictions, targets.unsqueeze(1))  # Ensure target shape matches prediction
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     print(f'Epoch {epoch}, Loss: {total_loss}')


















##################################Thursday
# prepping data and adding in padding V1 

# class CustomDataset(Dataset):
#     def __init__(self, dict_df, numeric_columns, target_column):
#         # Assuming numeric data can be safely converted to a NumPy array (check if this assumption holds true)
#         self.numeric_data = np.array([dict_df[col] for col in numeric_columns], dtype=np.float32).T  # Transpose to align with the number of samples
#         self.target_data = np.array(dict_df[target_column], dtype=np.float32)
        
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

# ###########
# # Dynamic Padding Sequence 
# from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch):
#     numeric_data = torch.stack([item['numeric_data'] for item in batch])
#     targets = torch.tensor([item['target'] for item in batch], dtype=torch.float)

#     # Dynamic padding for your multi-word categorical columns
#     observedPropertyDeterminandCode = pad_sequence([torch.tensor(item['observedPropertyDeterminandCode']) for item in batch], batch_first=True, padding_value=0)
#     resultUom = pad_sequence([torch.tensor(item['resultUom']) for item in batch], batch_first=True, padding_value=0)

#     return numeric_data, observedPropertyDeterminandCode, resultUom, targets

# # Adjusted DataLoader initialization without 'mappings' argument
# dataloader = DataLoader(CustomDataset(dict_df, numeric_columns, target_column), batch_size=32, shuffle=True, collate_fn=collate_fn)

################ here up okay i think, now need to figure out the model ##### 

# model suggestion - 4/2
class MyModel(nn.Module):
    def __init__(self, num_numeric_features, embedding_sizes):
        super(MyModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        # Assuming one embedding layer per categorical variable, `embedding_sizes` is a list of tuples:
        # [(num_categories_variable_1, embedding_dim_1), ..., (num_categories_variable_n, embedding_dim_n)]
        
        # Example for numeric and embedding combination
        self.fc_layers = nn.Sequential(
            nn.Linear(num_numeric_features + sum([size for _, size in embedding_sizes]), 128),  # Adjust sizes accordingly
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_numeric, x_categorical):
        embeddings = [embedding(cat) for embedding, cat in zip(self.embeddings, x_categorical)]
        x = torch.cat([x_numeric] + embeddings, dim=1)
        return self.fc_layers(x)





# Start below here on MOnday 
# getting the following error when running the below RuntimeError: Tensors must have same number of dimensions: got 2 and 3

#############
# NN MODEL 
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, numeric_size, embedding_dims, vocab_sizes):
        super(CustomModel, self).__init__()
        # Embedding layers for categorical data
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) 
            for vocab_size, embedding_dim in zip(vocab_sizes, embedding_dims)
        ])
        # Linear layer for numeric data
        self.numeric_layer = nn.Linear(numeric_size, 128)
        # Final layer after concatenating embeddings and numeric features
        self.final_layer = nn.Linear(128 + sum(embedding_dims), 1)

    def forward(self, numeric_data, observedPropertyDeterminandCode, resultUom):
        cat_embeddings = torch.cat([self.embeddings[0](observedPropertyDeterminandCode), 
                                    self.embeddings[1](resultUom)], dim=1)
        numeric_features = F.relu(self.numeric_layer(numeric_data))
        combined_features = torch.cat((numeric_features, cat_embeddings), dim=1)
        output = torch.sigmoid(self.final_layer(combined_features))
        return output


##################################working the padding section above 
# since we are using collate_fn we dont need to use context size 

# Initialize the model, loss function, and optimizer
# Corrected model initialization without context_sizes
model = CustomModel(numeric_size=len(numeric_columns), embedding_dims=[10, 10], vocab_sizes=[len(vocab1), len(vocab2)])
NUM_EPOCHS = 10  # Define the number of epochs for training


########## adjust here if needed ####
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001) #Optimizer Alternatives: If after some initial experiments you find that SGD isn't performing well, consider switching to optim.Adam for its adaptive learning rate capabilities:

# Training loop
# Assume NUM_EPOCHS is defined
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        numeric_data, observedPropertyDeterminandCode, resultUom, targets = batch
        model.zero_grad()
        # Adjust the model call according to its forward method's expected arguments
        output = model(numeric_data, observedPropertyDeterminandCode, resultUom)
        loss = loss_function(output.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


#################### working here up ##########










#########DATAFRAME VERSION ##########BELOW WED. 

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
df.drop('procedureAnalysedMedia', axis=1, inplace=True) # drop our target variable from the df 
df.columns # check 

##########################
# categorical columns identified and put into a list 
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# Define the embedding layer for the numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
target_column = 'procedureAnalysedMedia'

#
# Assuming df is your DataFrame
# Example preprocessing for numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])



##### here start down now, up above okay #####
# columns observedPropertyDeterminandCode , resultUom to be treated like they need to be word embedded 


# Convert categorical columns to 'category' dtype
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    df[col] = df[col].astype('category')








#############################working here down ############
# Attempt 2 
# Tokenize and encode categorical columns with sequences of words
CONTEXT_SIZE = 3  # Define a fixed context size for simplicity
def tokenize_and_encode(column, context_size=CONTEXT_SIZE):
    vocab = set(word for row in df[column].dropna() for word in row.split())
    word_to_ix = {word: i+1 for i, word in enumerate(vocab)}  # Start indexing from 1 to reserve 0 for padding
    
    # Tokenize and adjust sequences according to the context size
    encoded_sequences = []
    for row in df[column].fillna(''):
        words = row.split()
        encoded_sequence = [word_to_ix[word] if word in word_to_ix else 0 for word in words[:context_size]]
        encoded_sequence += [0] * (context_size - len(encoded_sequence))  # Padding
        encoded_sequences.append(encoded_sequence)
        
    return encoded_sequences, len(vocab) + 1  # +1 to account for padding token

encoded_observedPropertyDeterminandCode, vocab_size_observedPropertyDeterminandCode = tokenize_and_encode('observedPropertyDeterminandCode')
encoded_resultUom, vocab_size_resultUom = tokenize_and_encode('resultUom')

# Add encoded sequences back to DataFrame
df['encoded_observedPropertyDeterminandCode'] = encoded_observedPropertyDeterminandCode
df['encoded_resultUom'] = encoded_resultUom

# Update categorical columns list to include only the encoded versions
categorical_columns = ['encoded_observedPropertyDeterminandCode', 'encoded_resultUom']

class CustomDataset(Dataset):
    def __init__(self, dataframe, target_column, numeric_columns, categorical_columns):
        self.numeric_data = dataframe[numeric_columns].values.astype(np.float32)
        self.categorical_data = np.array(dataframe[categorical_columns].tolist()).astype(np.int64)
        self.targets = dataframe[target_column].values.astype(np.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'numeric_data': torch.tensor(self.numeric_data[idx], dtype=torch.float),
            'categorical_data': torch.tensor(self.categorical_data[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }
class CustomModel(nn.Module):
    def __init__(self, numeric_size, embedding_sizes, vocab_sizes):
        super(CustomModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for vocab_size, embedding_size in zip(vocab_sizes, embedding_sizes)])
        total_embedding_size = sum(embedding_sizes)
        
        self.linear1 = nn.Linear(total_embedding_size + numeric_size, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, numeric_data, categorical_data):
        cat_embeddings = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.embeddings)]
        cat_embeddings_combined = torch.cat(cat_embeddings, dim=1)
        combined = torch.cat([cat_embeddings_combined, numeric_data], dim=1)
        
        out = F.relu(self.linear1(combined))
        out = torch.sigmoid(self.linear2(out))
        return out

# Example embedding sizes for each categorical column (adjust as needed)
embedding_sizes = [10, 10]  # Example: 10-dimensional embeddings for both columns
vocab_sizes = [vocab_size_observedPropertyDeterminandCode, vocab_size_resultUom]

model = CustomModel(len(numeric_columns), embedding_sizes, vocab_sizes)
# Initialize your dataset and data loader
dataset = CustomDataset(df, 'procedureAnalysedMedia', numeric_columns, categorical_columns)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):  # Example: 10 epochs
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch['numeric_data'], batch['categorical_data'])
        loss = loss_function(output.squeeze(), batch['target'])
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

















#############ATTEMPT 1 BELOW#################
# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_column, numeric_columns, categorical_columns):
        self.dataframe = dataframe
        self.target = dataframe[target_column].values.astype(np.float32)
        self.numeric_data = dataframe[numeric_columns].values.astype(np.float32)
        self.categorical_data = np.stack([dataframe[col].cat.codes.values for col in categorical_columns], 1).astype(np.int64)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.numeric_data[idx], dtype=torch.float),
            torch.tensor(self.categorical_data[idx], dtype=torch.long),
            torch.tensor(self.target[idx], dtype=torch.float)
        )


# Neural Network Model
class CustomModel(nn.Module):
    def __init__(self, numeric_size, embedding_dims, vocab_sizes):
        super(CustomModel, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for vocab_size, embedding_dim in zip(vocab_sizes, embedding_dims)
        ])
        total_embedding_size = sum(embedding_dims)
        
        self.linear1 = nn.Linear(total_embedding_size + numeric_size, 128)
        self.linear1 = nn.Linear(embedding_dim+(context_size * embedding_dim), 128)

        self.linear2 = nn.Linear(128, 1)

    def forward(self, numeric_data, cat_data):
        embeddings = [embedding(cat_data[:, i]) for i, embedding in enumerate(self.embeddings)]
        cat_embeddings = torch.cat(embeddings, dim=1)
        combined = torch.cat([cat_embeddings, numeric_data], dim=1)
        
        out = torch.relu(self.linear1(combined))
        out = torch.sigmoid(self.linear2(out))
        return out









# Tokenizing sequences in cat5
vocab = set(word for row in df['cat5'] for word in row.split())
word_to_ix = {word: i for i, word in enumerate(vocab)}
# Convert sequences to numerical format
df['cat5'] = df['cat5'].apply(lambda x: [word_to_ix[word] for word in x.split()])


class CustomModel(nn.Module):
    def __init__(self, numeric_size, vocab_size, embedding_dim, num_categories, category_sizes):
        super(CustomModel, self).__init__()
        # Assuming embeddings for cat5
        self.embedding_cat5 = nn.Embedding(vocab_size, embedding_dim)
        # Embeddings for other categorical variables if necessary
        self.embeddings_cat = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in category_sizes])
        # Adjust the size for linear layers accordingly
        self.linear1 = nn.Linear(numeric_size + embedding_dim * (1 + len(category_sizes)), 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, numeric_data, cat_data, cat5_data):
        cat5_embed = self.embedding_cat5(cat5_data).view((1, -1))
        cat_embeds = [embed(cat_data[:, i]) for i, embed in enumerate(self.embeddings_cat)]
        embeds_full = torch.cat([cat5_embed] + cat_embeds + [numeric_data], -1)
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        return torch.sigmoid(out)


class CustomDataset(Dataset):
    def __init__(self, dataframe, target_col):
        self.dataframe = dataframe
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Assuming 'cat5' is the last categorical column with sequences
        cat_data = torch.tensor(row[['cat1', 'cat2', 'cat3', 'cat4']].values, dtype=torch.long)
        cat5_data = torch.tensor(row['cat5'], dtype=torch.long)
        numeric_data = torch.tensor(row[numeric_cols].values, dtype=torch.float)
        target = torch.tensor(row[self.target_col], dtype=torch.float).unsqueeze(-1)
        return numeric_data, cat_data, cat5_data, target




























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