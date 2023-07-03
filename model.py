from transformers import AutoModelForTokenClassification
from transformers import pipeline
from transformers import BertTokenizerFast 


class Model(object):
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained('tokenizer')
        self.model = AutoModelForTokenClassification.from_pretrained("artifacts")
        self.nlp_pipe = pipeline("ner", model=self.model, tokenizer=self.tokenizer)


    def predict(self, input):
        """token classification"""
        result = None
        try:
            result = self.nlp_pipe(input)
        except Exception as ex:
            print("Unable to predict")

        return result
        