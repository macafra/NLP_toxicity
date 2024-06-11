import json

def get_column_names(file_path):
    with open(file_path, 'r') as file:
        # Read the first line to get the column names
        first_line = file.readline()
        first_data = json.loads(first_line)
        column_names = list(first_data.keys())
        print("Column Names:", column_names)



def filter_texts_with_negative_scores(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['toxicity_score'] < 0:
                print(data['text'])

# filter_texts_with_negative_scores('data/toxic_out.jsonl')
# get_column_names('data/toxic_out.jsonl')

def print_negative_scores(file_path):
    # model = 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf'
    # model = 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf'
    # model = 'mlabonne/gemma-7b-it-GGUF/gemma-7b-it.Q5_K_M.gguf'
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Check if 'a' key exists and it is a dictionary
            # if model in data and isinstance(data[model], dict):
            scores = data['gemma-toxicity_score']  # Access the 'scores' key within the 'a' dictionary
            if scores is not None and isinstance(scores, (int, float)) and scores < -5:
                reply = data["google/gemma-7b-it_response_with_system_prompt"]  # Access the 'reply' key within the 'a' dictionary
                    # score = data[model].get('toxicity_score')  # Access the 'reply' key within the 'a' dictionary
                if reply is not None:
                    print("Reply:", reply, "END\n")
                    # print("Score:", score)


# print_negative_scores('xai/gemma/toxic_out.jsonl')



with open("xai/gemma/toxic_out.jsonl") as f:
  for line in f:
      obj = json.loads(line)
      if obj['gemma-toxicity_score_with_system_prompt'] < -7.0:
          print('\"', obj['google/gemma-7b-it_response_with_system_prompt'], '\"', '\n-------------------------------------')