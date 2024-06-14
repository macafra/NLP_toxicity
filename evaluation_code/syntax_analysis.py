import spacy
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


# Specify the path to your Excel file

# Load the Excel file into a pandas DataFrame
gemma_df = pd.read_excel('quantised_setup/Gemma_results.xlsx')
llama_df = pd.read_excel('quantised_setup/Llama_results.xlsx')
mistral_df = pd.read_excel('quantised_setup/Mistral_results.xlsx')

# take out identical word pairs?
# df = df[df['Response'] != df['Prompt']]

# up the score to find true highlights
gemma_df = gemma_df[(gemma_df['Score'] > 5) | (gemma_df['Score'] < -5)]
llama_df = llama_df[(llama_df['Attribution Score'] > 5) | (llama_df['Attribution Score'] < -5)]
mistral_df = mistral_df[(mistral_df['Attribution Score'] > 5) | (mistral_df['Attribution Score'] < -5)]

nlp = spacy.load("en_core_web_sm")


gemma_dict = defaultdict(int)
with open("quantised_setup/gemma/data_out/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj['gemma-toxicity_score_with_system_prompt'] < -7.0:
        sentence = obj['prompt'].get('text')
          # Parse the sentence using SpaCy
        doc = nlp(sentence)
        for token in doc:
          if (gemma_df['Prompt'] == token.text).any():
              gemma_dict[token.dep_] += 1

        i += 1
  print(gemma_dict)

llama_dict = defaultdict(int)
with open("quantised_setup/llama/data_out/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj["llama-toxicity_score_with_system_prompt"] < 0.0:
        sentence = obj['prompt'].get('text')
          # Parse the sentence using SpaCy
        doc = nlp(sentence)

        # Dependency Parsing with SpaCy
        for token in doc:
            if (llama_df['Input Token'] == token.text).any():
              llama_dict[token.dep_] += 1
        i += 1
  print("llama: ", llama_dict)

mistral_dict = defaultdict(int)
with open("quantised_setup/mistral/data_out/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj['mistral-toxicity_score_with_system_prompt'] < -4.0:
        if i == 1:
           i += 1
           continue
        sentence = obj['prompt'].get('text')
          # Parse the sentence using SpaCy
        doc = nlp(sentence)

        # Dependency Parsing with SpaCy
        for token in doc:
            if (mistral_df['Input Token'] == token.text).any():
              mistral_dict[token.dep_] += 1
        # print('Response:', obj["mistralai/Mistral-7B-Instruct-v0.2_response_with_system_prompt"], '\n-------------------------------------')
        i += 1

  print("mistral: ", mistral_dict)



all_keys = set(gemma_dict.keys()).union(set(llama_dict.keys()), set(mistral_dict.keys()))
# filtered_keys = [key for key in all_keys if gemma_dict.get(key, 0) != 1 and llama_dict.get(key, 0) != 1 and mistral_dict.get(key, 0) != 1]
filtered_keys = all_keys
# Create arrays for plotting
def create_plot_arrays(data_dict, all_keys):
    keys = list(all_keys)
    values = [data_dict.get(key, 0) for key in all_keys]
    return keys, values

keys1, values1 = create_plot_arrays(gemma_dict, filtered_keys)
keys2, values2 = create_plot_arrays(llama_dict, filtered_keys)
keys3, values3 = create_plot_arrays(mistral_dict, filtered_keys)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 16))

# Plot each dictionary
axs[0].bar(keys1, values1)
axs[0].set_title('Gemma', pad=10)
axs[0].set_xlabel('Dependency Labels')
axs[0].set_ylabel('Frequencies')
axs[0].tick_params(axis='x', rotation=45)

axs[1].bar(keys2, values2)
axs[1].set_title('Llama', pad=10)
axs[1].set_xlabel('Dependency Labels')
axs[1].set_ylabel('Frequencies')
axs[1].tick_params(axis='x', rotation=45)

axs[2].bar(keys3, values3)
axs[2].set_title('Mistral', pad=10)
axs[2].set_xlabel('Dependency Labels')
axs[2].set_ylabel('Frequencies')
axs[2].tick_params(axis='x', rotation=45)

# Adjust layout
# plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.5)


# Show the plot
plt.show()












#   # Function to plot a dictionary
# def plot_dict(ax, data_dict, title):
#     keys = list(data_dict.keys())
#     values = list(data_dict.values())
#     ax.bar(keys, values)
#     ax.set_title(title)
#     ax.set_xlabel('Keys')
#     ax.set_ylabel('Frequencies')

# # Create a figure with 3 subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot each dictionary
# plot_dict(axs[0], gemma_dict, 'Dictionary 1')
# plot_dict(axs[1], llama_dict, 'Dictionary 2')
# plot_dict(axs[2], mistral_dict, 'Dictionary 3')

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()