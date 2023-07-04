# BERT Based NER API
This repo contains NER model implementation for Conll2003 dataset using a transformer-based model for predicting person, organization etc from the input text, For Inferenceing Part Fast API is used.

### Step -1 Fine Tuning
Run the Training Notebook provided, if you don't have a local resource, it's suggested to run either in Colab or Kaggle.

`Run training-notebook.ipynb`

After a successful run, the output contains the fine-tuned model along with JSON config file and tokenizer files.

#### Note: 
 The fine-tuned model and JSON config file should be placed in the artifacts folder(by default it keeps in the artifacts folder). And tokenizer-related files should be placed in the tokenizer folder.

### Step - 2 Prerequisites
Install Required packages by executing the below command

`pip install -r requirements.txt`

### Step - 3 Start API
Start the server by executing the below command.

`sh start_server.sh`

### Step - 4 Swagger UI
Go to localhost:8001/docs, you should be able to see the Swagger UI. Where you can test the API.

## Features
- BERT Base Model is used for NER.
- FastAPI is used for Inferencing.
- Hugging Face Library is used.
- Supports Electra Model without any code changes.
- Dataset will get downloaded automatically using Hugging Face library.


## License
MIT
