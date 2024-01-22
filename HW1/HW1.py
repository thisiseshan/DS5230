#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 23:34:23 2024

@author: eshan
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#%%
#* --- paperTitle
#@ --- Authors
#t ---- Year
#c  --- publication venue
#index 00---- index id of this paper
#% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
#! --- Abstract

# Problem 1
data_dict = {}
record = []
ref = []
references = set()
title = set()
author_set = set()
publication_venue = set()

file_path = '/Users/eshan/Documents/DS 5230/HW1/datasets/acm.txt'
total_lines = sum(1 for line in open(file_path, 'r'))

with open(file_path) as file:
    
    i = 0
    for line in tqdm(file, total=total_lines):
        
        if line[:2] == "#*":
            count = 1
            data_dict['title'] = line[2:].strip()
            title.add(line[2:].strip())
        elif line[:2] == "#@":

            data_dict['authors'] = line[2:].strip()
            for auth in line[2:].strip().split(','):
                author_set.add(auth)        
        elif line[:2] == "#t":
            data_dict['year'] = line[2:].strip()
        elif line[:2] == "#c":
            data_dict['publication_venue'] = line[2:].strip()
            publication_venue.add(line[2:].strip())
        elif line[:6] == "#index":
            data_dict['index'] = line[6:].strip()
        elif line[:2] == "#%":
            ref.append(line[2:-1])
            references.add(line[2:-1])
            data_dict['references'] = ref
        elif line[:2] == "#!":
            data_dict['abstract'] = line[2:]
        elif line == '\n' and count==1:
            record.append(data_dict)
            ref = []
            data_dict = {}
            count = 2
#%%
df = pd.DataFrame(record)
#%%
df.head(20)
#%%
# Question 1: A) Compute the number of distinct authors, publication venues,
# publications, and citations/references

len(title)
#%%
len(author_set)
#%%
len(publication_venue)
#%%
len(references)
#%%
# B. Are these numbers likely to be accurate?
# As an example look up all the publications venue names associated with the conference
# “Principles and Practice of Knowledge Discovery in Databases” – what do you notice?

df.loc[df['publication_venue'].str.contains('Principles and Practice of Knowledge Discovery in Databases', na=False)]

# 200 rows: PKDD and PKDD\' are different
#%%
#C:  For each author, construct the list of publications.
# Plot a histogram of the number of publications per author (use a logarithmic scale on the y axis)

author_dict = {}

for index, row in df.iterrows():

    if pd.isna(row['authors']):
        continue
    else:
        for auth in row['authors'].split(','):
            if auth in author_dict:
                author_dict[auth].append(row['title'])
            else:
                author_dict[auth] = [row['title']]

                # authors_set.add(row['authors'])
#%%

author_dict
new_count_dict = {}

for key in author_dict.keys():
    new_count_dict[key] = len(author_dict[key])
#%%
# C.  For each author, construct the list of publications. Plot a histogram of
# the number of publications per author (use a logarithmic scale on the y axis)
new_count_dict
publication_counts = list(new_count_dict.values())

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(publication_counts, bins=range(1, 2001), log=True)  # Bins


# Adding titles and labels
plt.title('Histogram of Publication Counts')
plt.xlabel('Number of Publications')
plt.ylabel('Number of Authors (log scale)')
# # Show the plot
plt.xscale('linear')
plt.yscale('log')
plt.tight_layout()
plt.show()

# print(publication_counts)
#%%
# D. Calculate the mean and standard deviation of the number of publications per author.
# Also calculate the Q1 (1st quartile14), Q2 (2nd quartile, or median) and Q3 (3rd quartile) values.
# Compare the median to the mean and explain the difference between the two values based on the 
# standard deviation and the 1st and 3rd quartiles.

# mean

publications = pd.Series(list(new_count_dict.values()))
          
print("Mean and Std")               
print(publications.mean())
print(publications.std())

print("Quantile1, Median, Quantile3") 
print(publications.quantile(0.25))      
print(publications.median())       
print(publications.quantile(0.75))       

# For comparison we have mean as 16 and median as 2 so there the graph is heavily right skewed. 
# Most authors have small number of publications
# First quantile = median, means that at least 50% of the authors have 2 or fewer publications
# std of 25 indicates that there are authors with publication counts very far from the mean (outliers)         
#%%
# E. Now plot a histogram of the number of publications per venue,
# as well as calculate the mean, standard deviation, median, Q1, and Q3 values. 
# What is the venue with the largest number of publications in the dataset?
venue_dict = {}

for index, row in df.iterrows():

    if pd.isna(row['publication_venue']):
        continue
    else:
        if row['publication_venue'] in venue_dict:
            venue_dict[row['publication_venue']].append(row['title'])
        else:
            venue_dict[row['publication_venue']] = [row['title']]

# plt.figure(12,6)
# plt.hist()
venue_dict

#%%
venue_dict_count = {}


for venue in venue_dict:
    venue_dict_count[venue] = len(venue_dict[venue])

# authors_set
#%%
venue_dict_count
#%%
plt.figure(figsize=(12,6))

plt.hist(list(venue_dict_count.values()),bins=range(1,2000), log=True)
plt.title('Venue based publications')
plt.xlabel('Number of publications')
plt.ylabel('Number of venues')
plt.tight_layout()
plt.show()

#%%
venue_counter = list(venue_dict_count.values())
venue_df = pd.Series(venue_counter)

print(venue_df.mean())
print(venue_df.std())
print(venue_df.median())
print(venue_df.quantile(0.25))
print(venue_df.quantile(0.75))
#%%
# print(dict(sorted(venue_dict_count.items(), key=lambda item:item[1])))
print(max(venue_dict_count, key=venue_dict_count.get))

#%%
# F. Plot a histogram of the number of references (number of publications a publication refers to)
# and citations (number of publications referring to a publication) per publication. 
# What is the publication with the largest number of references? 
# What is the publication with the largest number of citations? Do these make sense?

reference_count_dict = {}

for index, row in df.iterrows():
    if pd.isna(row['title']):
        continue
    
    # if pd.isna(df['references']).any():
    #     reference_count_dict[row['title']] = 0
    # else:
    #  
    if type(row['references']) == list:
        reference_count_dict[row['title']] = len(row['references'])
#%%
reference_count_dict
#%%
plt.figure(figsize=(12,6))
plt.hist(list(reference_count_dict.values()),bins=range(1,1000), log=True)

plt.title('Number of references')
plt.xlabel('Number of references')
plt.ylabel('Number of publications')
plt.show()

#%%
df = df.dropna(subset=['index'])
df['index'] = df['index'].astype(str).str.strip()

citation_count_dict = {row['title']: 0 for index, row in df.iterrows()}


reference_dict = {}
for index, row in df.iterrows():
    if isinstance(row['references'], list):
        for ref in row['references']:
            ref = ref.strip()  # Clean reference string
            if ref in reference_dict:
                reference_dict[ref].add(row['index'])
            else:
                reference_dict[ref] = {row['index']}

#%%
# Count citations
for index, row in df.iterrows():
    count = len(reference_dict.get(row['index'], []))
    citation_count_dict[row['title']] = count

#%%
citation_count_dict 
#%%

plt.figure(figsize=(12,6))
plt.hist(citation_count_dict.values(), bins=range(1,1000), log=True)

plt.title('Citations per publication')
plt.xlabel('Citations')
plt.ylabel('Publications')
plt.show()
#%%
# G. Calculate the so called “impact” factor for each venue.
# To do so, calculate the total number of citations for the publications in the venue,
# and then divide this number by the number of publications for the venue. Plot a histogram of the results

# Citations per venue
from collections import defaultdict

# Create a mapping from index to publication venue
index_to_venue = pd.Series(df['publication_venue'].values, index=df['index']).to_dict()

# Initialize a defaultdict to count citations for each venue
citation_count = defaultdict(int)

for references in df['references'].dropna():
    if isinstance(references, list):
        for ref in references:
            ref_venue = index_to_venue.get(ref.strip())
            if ref_venue:
                citation_count[ref_venue] += 1

# Convert defaultdict to a regular dict for the final result
citation_count_dict2 = dict(citation_count)

#%%
citation_count_dict2
#%%
# Total number of Publications per venue

# df.groupby('publication_venue')['title'].sum().size()
publication_count_per_venue = {}

for index, row in df.iterrows():
    if isinstance(row['publication_venue'], str):
        venue = row['publication_venue'].strip()  # Clean venue string
        if venue in publication_count_per_venue:
            publication_count_per_venue[venue] += 1  # Increment count
        else:
            publication_count_per_venue[venue] = 1

#%%
publication_count_per_venue

#%%
impact = {}

for key in publication_count_per_venue.keys():
    if key in citation_count_dict2.keys():
        impact[key] = citation_count_dict2[key] / publication_count_per_venue[key]
#%%
plt.figure(figsize=(12,6))
plt.hist(impact.values(), bins=20, log=True)

plt.title('Impact factor')
plt.xlabel('Impact')
plt.ylabel('Publications')
plt.show()
#%%
# H. What is the venue with the highest apparent impact factor? Do you believe this number?
# (http://mdanderson.libanswers.com/faq/26159)

max(impact.values())

# I dont believe this number, 80000 is bonkers
#%%
# I. Now repeat the calculation from item G, but restrict the calculation to venues with at least 10 publications.
# How does your histogram change? List the citation counts for all publications from the venue with the highest impact factor.
# How does the impact factor (mean number of citations) compare to the median number of citations?


impact2 = {}

for key in publication_count_per_venue.keys():
    if key in citation_count_dict2.keys():
        if publication_count_per_venue[key] >= 10:
            impact2[key] = citation_count_dict2[key] / publication_count_per_venue[key]

#%%
plt.figure(figsize=(12,6))
plt.hist(impact2.values(), bins=20, log=True)

plt.title('Impact factor')
plt.xlabel('Impact')
plt.ylabel('Publications')
plt.show()

# The graph is more reasonable and the values are much lower - still presence of outlier.
#%%
sorted(impact2.items(), key=lambda x: x[1], reverse=True)[0:10]

#%%
# Mean vs median impact factor
import statistics

statistics.mean(list(impact2.values()))
statistics.median(list(impact2.values()))

# Much higher mean, the distribution is right skewed.
#%%
# J inally, construct a list of publications for each publication year. 
# Use this list to plot the average number of references and average number of citations per publication as a function of time. 
# Explain the differences you see in the trends.

df2 = df.dropna(subset=['year'])
df2 = df2.groupby('year')['index'].count().reset_index()

publication_per_year = {}
df2

for index,row in df2.iterrows():
    publication_per_year[row['year']] = row['index']
#%%
publication_per_year
#%%
df.columns
#%%
# references per year
references_per_year = {}

for index,row in df.iterrows():
    if isinstance(row['references'], list):
        references_per_year[row['year']] = len(row['references'])
#%%
references_per_year
#%%
index_year_mapping = pd.Series(df['year'].values, index=df['index']).to_dict()

#%%
index_year_mapping
#%%
count_citation_year = defaultdict(int)
df = df.dropna(subset=['year'])

for references in df['references'].dropna():
    if isinstance(references, list):
        for ref in references:
            ref_year = index_year_mapping.get(ref.strip())
            if ref_year:
                count_citation_year[ref_year] +=1
        
citation_year_dict = dict(count_citation_year)
#%%

citation_year_dict = {k: v for k, v in citation_year_dict.items() if not (isinstance(k, float) and math.isnan(k))}
#%%
citation_per_year = citation_year_dict

#%%
citation_per_year
#%%
references_per_year
#%%
publication_per_year
#%%
avg_references_per_pub = {year: references_per_year[year]/publication_per_year[year] 
                          for year in references_per_year}
avg_citations_per_pub = {year: citation_per_year[year]/publication_per_year[year] 
                         for year in citation_per_year}
#%%
# Plotting
years = sorted(avg_references_per_pub.keys())  # Sorted list of years
avg_refs = [avg_references_per_pub[year] for year in years]
avg_cites = [avg_citations_per_pub[year] for year in years]

plt.figure(figsize=(10, 5))
plt.plot(years, avg_refs, label='Average References per Publication')
plt.plot(years, avg_cites, label='Average Citations per Publication')
plt.xlabel('Year')
plt.ylabel('Average Count')
plt.title('Average References and Citations per Publication Over Time')
plt.legend()
plt.show()















#%%






