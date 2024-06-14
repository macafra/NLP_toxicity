import spacy
import json

nlp = spacy.load("en_core_web_sm")



with open("quantised_setup/gemma/data_out/toxic_out.jsonl") as f:
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


with open("quantised_setup/llama/data_out/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj["llama-toxicity_score_with_system_prompt"] < 0.0:
        print(i)
        sentence = obj['prompt'].get('text')
          # Parse the sentence using SpaCy
        doc = nlp(sentence)

        # Dependency Parsing with SpaCy
        print("Dependency Parsing:")
        for token in doc:
            print(f"{token.text:<10}{token.dep_:<10}{token.head.text:<10}{[child for child in token.children]}")

        print('Response:', obj["meta-llama/Meta-Llama-3-8B-Instruct_response_with_system_prompt"], '\n-------------------------------------')
        i += 1


with open("quantised_setup/mistral/data_out/toxic_out.jsonl") as f:
  i = 1
  for line in f:
      obj = json.loads(line)
      if obj['mistral-toxicity_score_with_system_prompt'] < -4.0:
        if i == 1:
           i += 1
           continue
        print(i)
        sentence = obj['prompt'].get('text')
        print(sentence)
          # Parse the sentence using SpaCy
        doc = nlp(sentence)

        # Dependency Parsing with SpaCy
        print("Dependency Parsing:")
        for token in doc:
            print(f"{token.text:<10}{token.dep_:<10}{token.head.text:<10}{[child for child in token.children]}")

        print('Response:', obj["mistralai/Mistral-7B-Instruct-v0.2_response_with_system_prompt"], '\n-------------------------------------')
        i += 1