# NER API
This API takes text as input and provides the token classification(NER) as output. To run the API few steps need to be performed.

### Step -1
Run the Training Notebook provided, if you don't have a local resource, it's suggested to run either in colab or Kaggle.

`Run traning-notebook.ipynb`

After successful run, the output contains the fine-tuned model along with JSON config file and tokenizer files.

#### Note:
 The fine-tuned model and JSON config file should be placed in the artifacts folder(by default it keeps in the artifacts folder). And tokenizer-related files should be placed in the tokenizer folder.

### Step - 3
Install Required packages by executing the below command

`pip install -r requirements.txt`

### Step - 4
Start the server by executing the below command.
`sh start_server.sh`

### Step - 5 
Go to localhost:8001/doc, you should be able to see the Swagger UI. Where you can test the API.

## Features
- BERT Base Model is used for NER.
- FastAPI is used for Inferencing.
- Hugging Face Library is used.
- Supports Electra Model without any code changes.
- Dataset will get downloaded automatically using Hugging Face library.


## License
MIT
