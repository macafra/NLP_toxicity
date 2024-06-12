import spacy
# import nltk
# from nltk import pos_tag, ne_chunk
# from nltk.tokenize import word_tokenize
# from nltk.tree import Tree
import json

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# # Example sentence
# sentence = "She enjoys reading books on rainy days."

# # Parse the sentence using SpaCy
# doc = nlp(sentence)

# # Dependency Parsing with SpaCy
# print("Dependency Parsing:")
# for token in doc:
#     print(f"{token.text:<10}{token.dep_:<10}{token.head.text:<10}{[child for child in token.children]}")

# # Visualize the dependency parse
# spacy.displacy.render(doc, style="dep", jupyter=False)

# Constituent Parsing with NLTK
# def nltk_tree_to_string(tree):
#     """
#     Convert an nltk Tree to a string format for easier viewing.
#     """
#     if isinstance(tree, Tree):
#         return f"({tree.label()} {' '.join(nltk_tree_to_string(child) for child in tree)})"
#     else:
#         return tree

# # Tokenize and POS tag the sentence for NLTK
# tokens = word_tokenize(sentence)
# tagged_tokens = pos_tag(tokens)

# # Perform named entity recognition
# ne_tree = ne_chunk(tagged_tokens)

# # Convert NLTK NE tree to a string
# constituent_parse = nltk_tree_to_string(ne_tree)
# print("\nConstituent Parsing:")
# print(constituent_parse)


with open("xai/gemma/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj['gemma-toxicity_score_with_system_prompt'] < -7.0:
        print(i)
        sentence = obj['prompt'].get('text')
          # Parse the sentence using SpaCy
        doc = nlp(sentence)

        # Dependency Parsing with SpaCy
        print("Dependency Parsing:")
        for token in doc:
            print(f"{token.text:<10}{token.dep_:<10}{token.head.text:<10}{[child for child in token.children]}")

        print('Response:', obj['google/gemma-7b-it_response_with_system_prompt'], '\n-------------------------------------')
        i += 1