#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:34:54 2024

@author: eshan
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import StringIO
#%%
file_path = '/Users/eshan/Documents/DS 5230/HW1/datasets/kosarak.dat.txt'
total_lines = sum(1 for line in open(file_path, 'r'))
#%%

print(total_lines)

#%%
rows = 990003 # Number of rows
cols = 41271  # Number of columns

# Create the matrix filled with zeros
matrix = np.zeros((rows, cols))
#%%

matrix
#%%

with open(file_path, 'r') as file:
    row_index = 0
    for line in file:
        transaction = list(map(int, line.strip().split()))
        for i in transaction:
            matrix[row_index][i] = 1
        row_index+=1
        # print(transaction)
 
        # transactions.append(transaction)
        # unique_ids.update(transaction)
        
#%%
# matrix = matrix[:, 1:]
# matrix = matrix.astype(np.int8)
df=pd.DataFrame(matrix)
df.to_csv('matrix-og.csv')
#%%
df.head(10)

#%%
# df.to_csv('output.csv')
from scipy.sparse import lil_matrix
import pandas as pd

# file_path = '/path/to/your/file'
rows = 990003  # Number of rows
cols = 41271   # Number of columns - 1, since you're dropping the first column

# Use a sparse matrix
matrix = lil_matrix((rows, cols), dtype=np.int8)

with open(file_path, 'r') as file:
    for row_index, line in enumerate(file):
        transaction = [int(i) - 1 for i in line.strip().split()]  # Subtract 1 to account for dropped column
        matrix[row_index, transaction] = 1

# Converting to DataFrame (if still needed)
# Note: This step can be very memory-intensive with such a large matrix
# df = pd.DataFrame(matrix.toarray())
# #%%
# df.to_csv('output.csv')
#%%
def to_sparse_arff(sparse_matrix, num_columns, output_file_path):
    """ Convert the sparse matrix to ARFF format and write to a file. """
    with open(output_file_path, 'w') as file:
        file.write('@relation itemsets\n\n')
        
        # Write the attribute section
        for col in range(num_columns):
            file.write(f'@attribute item{col + 1} {{0, 1}}\n')
        
        file.write('\n@data\n')
        
        # Write the data section in sparse ARFF format
        for row in range(sparse_matrix.shape[0]):
            row_data = sparse_matrix.getrow(row).nonzero()[1]
            if row_data.size > 0:
                data_line = '{' + ', '.join(f'{col} 1' for col in row_data) + '}\n'
                file.write(data_line)
#%%
file_path = 'output.arff'  # Replace with your desired output file path
to_sparse_arff(matrix, cols, file_path)


#%%
weka_dict = {}
index = 0
maxi = 0
with open(file_path) as file:
    for line in tqdm(file):
        # transaction = set(map(int, line.strip().split()))
        # print(transaction)
        line = line.strip()
        line_list = [int(number) for number in line.split()]
        maxi = max(maxi, max(line_list))
        # print(max(line))
        # print(line)
        # print(line_list)
        # weka_dict[]
        # print(index)
        # break
å
#%%
unique_ids = set()
transactions = []

# Read file and populate transactions and unique_ids
with open(file_path, 'r') as file:
    for line in file:
        transaction = set(map(int, line.strip().split()))
        transactions.append(transaction)
        unique_ids.update(transaction)

# Sort the unique IDs for consistent column ordering
unique_ids = sorted(unique_ids)
#%%

unique_ids

#%%
import csv

output_file_path = 'output.csv'

# Open a file to write the binary matrix
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header (unique_ids)
    writer.writerow(unique_ids)
    for transaction in transactions:
        row = np.array([1 if uid in transaction else 0 for uid in unique_ids], dtype=np.int8)
        writer.writerow(row)



#%%

# Efficiently create binary data representation
binary_data = {uid: np.array([1 if uid in transaction else 0 for transaction in transactions], dtype=np.int8) 
               for uid in unique_ids}

#%%

# Convert the dictionary to a DataFrame
binary_df = pd.DataFrame(binary_data)

binary_df.head()  # Display the first few rows of the DataFrame

#%%
binary_df.head()
#%%
df = pd.read_csv('output.csv', nrows=10)
df.to_csv('sample.csv')
#%%
df.describe()
#%%
df['1'][0].type()
#%%
from tqdm import tqdm

def to_sparse_arff(sparse_matrix, num_columns, output_file_path):
    """ Convert the sparse matrix to ARFF format and write to a file. """
    with open(output_file_path, 'w') as file:
        file.write('@relation itemsets\n\n')
        
        # Write the attribute section
        for col in range(num_columns):
            file.write(f'@attribute item{col + 1} {{0, 1}}\n')
        
        file.write('\n@data\n')
        
        # Write the data section in sparse ARFF format
        for row in tqdm(range(sparse_matrix.shape[0]), desc="Writing rows"):
            row_data = sparse_matrix.getrow(row).nonzero()[1]
            if row_data.size > 0:
                data_line = '{' + ', '.join(f'{col} 1' for col in row_data) + '}\n'
                file.write(data_line)

# Usage
file_path = 'output2.arff'  # Replace with your desired output file path
to_sparse_arff(matrix, cols, file_path)







#%%


from scipy.sparse import lil_matrix

from scipy.sparse import lil_matrix

def convert_to_arff_optimized(dat_file_path, arff_file_path):
    # Read all transactions and find unique items
    unique_items = set()
    transactions = []
    
    with open(dat_file_path, 'r') as file:
        for line in file:
            transaction = set(map(int, line.strip().split()))
            transactions.append(transaction)
            unique_items.update(transaction)

    # Sort the items and map them to indices
    unique_items = sorted(unique_items)
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}

    # Create a sparse matrix
    sparse_matrix = lil_matrix((len(transactions), len(unique_items)), dtype=int)
    for row_idx, transaction in enumerate(transactions):
        for item in transaction:
            sparse_matrix[row_idx, item_to_index[item]] = 1

    # Convert to CSR format for efficient row slicing
    csr_matrix = sparse_matrix.tocsr()

    # Write to ARFF file with buffered writing
    with open(arff_file_path, 'w', buffering=1_000_000) as arff_file:
        arff_file.write('@relation transaction_data\n')
        for item in unique_items:
            arff_file.write(f'@attribute item{item} {{0, 1}}\n')
        arff_file.write('\n@data\n')

        # Optimized writing of sparse data
        for row in tqdm(range(csr_matrix.shape[0])):
            row_data = csr_matrix.getrow(row).nonzero()[1]
            if row_data.size > 0:
                data_line = '{' + ', '.join(f'{col} 1' for col in row_data) + '}\n'
                arff_file.write(data_line)



# Paths for the DAT and ARFF files
dat_file_path = '/Users/eshan/Downloads/kosarak.dat'
arff_file_path = 'kosarak2.arff'

convert_to_arff_optimized(dat_file_path, arff_file_path)






#%%
























