# NER API
This API takes text as input and provides the token classification(NER) as output. To run the API few steps need to performed.

### Step -1
1. Run the Training Notebook provided, if you dont have local resource, its suggested to run either in colab or kaggle.
<br>
`Run traning-notebook.ipynb in Jupyter notebook`
2. After succesfull run, the output contains the fine-tuned model along with json config file and tokenizer files.
Note: The fine-tuned model and json config file should be placed in artifacts folder(by default it keeps in artifacts folder). And tokenizer related files should be placed in tokenizer folder.

### Step - 3
1. Install Required packages by executing the below command
<br>
`pip install -r requirements.txt`

### Step - 4
1. Start the server by executing the below command.
`sh start_server.sh`

### Step - 5 
1. Go to localhost:8001/doc, you should be able to see the Swagger UI. Where you can test the API.

## Features
- BERT Base Model is used for NER.
- FastAPI is used for Inferencing.
- Hugging Face Library is used.
- Supports Electra Model without any code changes.
- Dataset will get downloaded automatically using Hugging Face library.


## License
MIT