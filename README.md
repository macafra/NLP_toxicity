# Characterising Toxicity in Language Models
## How to run the code
Steps needed before running the code:

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Add your Hugging Face access token to your environment variables as 'HF_TOKEN'.

The Jupiter notebooks for running our experiment can be found in the folder `quantised_setup`. In here are for all the models seperate notebooks for generating the output and then retrieving explanations (XAI). We looked at three models for our study: [Gemma-7b-it](https://huggingface.co/google/gemma-7b-it), [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). All of them come from Hugging Face.

For running our experiments, we used [Kaggle](https://www.kaggle.com) due to the resources available there (weekly GPU).

Additionally, there is a `lm_studio_setup` in which code is included to run the models with LM Studio. 

## File structure

In `quantised_setup`:
- `data_in`: this is the input data from the [DecodingTrust benchmark](https://decodingtrust.github.io).
- `gemma`: in this folder are the files: `analyse.ipynb`, `generate-google-gemma.ipynb`, `xai-google-gemma.ipynb` and the results.
- `llama`: in this folder are the files: `analyse.ipynb`, `generate-meta-llama.ipynb`, `xai-meta-llama.ipynb` and the results.
- `mistral`: in this folder are the files: `analyse.ipynb`, `generate-mistral.ipynb`, `xai-mistral.ipynb` and the results.
- `toxicity_score_calculation.ipynb`: this notebook is used to calculate the toxicity scores of the outputs from the models; [nicholasKluge/ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel) was used for this calculation.


### Used technologies:
* <a href="https://captum.ai" target="_blank">Captum</a>
* <a href="https://github.com/TimDettmers/bitsandbytes" target="_blank">bitsandbytes</a>
