# Required Imports
import datasets
import numpy as np
from transformers import BertTokenizerFast, ElectraTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer


class NERTraining:
    def __init__(self, model_name) -> None:
        self.model_name = model_name


    def get_dataset(self, dataset_name: str):
        """Downloads dataset from hugging face""" 
        dataset = None
        try: 
            dataset = datasets.load_dataset(dataset_name)
        except Exception as ex:
            print("Unable to download dataset - ", ex)

        return dataset

    def get_tokenizer(self):
        """Returns the tokenizer based on the model selected"""
        tokenizer = None
        try:
            if self.model_name.lower() == "bert-base-uncased":
                tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            elif "electra" in self.model_name:
                tokenizer = ElectraTokenizerFast.from_pretrained("bert-base-uncased")
        except Exception as ex:
            print("Unable to get tokenizer for the model -%s", self.model_name)

        return tokenizer


    def format_labels(self, data,  label_all=True):
        """
        Appends -100 for the None Type and returns the labels
        """
        tokenizer = self.get_tokenizer()
        tokenized_input = tokenizer(data['tokens'], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(data['ner_tags']):
            word_ids = tokenized_input.word_ids(batch_index=i)
            label_ids = []
            pre_ind = None

            for wi in word_ids:
                if wi is None:
                    label_ids.append(-100)
                elif wi != pre_ind:
                    label_ids.append(label[wi])
                else:
                    label_ids.append(label[wi] if label_all else -100)

                pre_ind = wi

            # now append to labels list
            labels.append(label_ids)

        tokenized_input['labels'] = labels

        return tokenized_input


    def get_model(self):
        """Returns the model instance"""
        model = None
        try:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=9)
        except Exception as ex:
            print("Unable to download the model - ",self.model_name)
          
        return model

    def set_arguments(self, m_args:dict):
        """Based on give settings create args object"""
        args = None
        try:
            args = TrainingArguments(**m_args)
        except Exception as ex:
            print("Unable to create args object based on the provided - ", ex)
        return args 

    def get_data_collator(self, tokenizer):
        """data collator """
        data_collator = None
        try:
            data_collator = DataCollatorForTokenClassification(tokenizer)
        except Exception as ex:
            print("Data collator operation failed - ", ex)

        return data_collator

    def get_metrics(self):
        metrics = None
        try:
           metrics = datasets.load_metric("seqeval")
        except Exception as ex:
            print("Unable to load metrics from seqeval - ", ex)
        return metrics

    def compute_metrics(self, p):
        """computest result for the prediction and actual output"""
        label_list = dataset['train'].features['ner_tags'].feature.names
        metrics = self.get_metrics()
        predictions, labels = p
        #select predicted index with maximum logit for each token
        predictions = np.argmax(predictions, axis=2)

        # model predictions
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # actual prediction
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # compute result
        results = metrics.compute(predictions=true_predictions, references=true_labels)

        result_dict =  {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        return result_dict

    def model_training(self, model, args, train_dataset, eval_dataset, data_collator, tokenizer, compute_metrics):
        """Trains the model based on give params"""
        try:
            trainer = Trainer(
                                model,
                                args,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                data_collator=data_collator,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metrics
                            )
            trainer.train()

        except Exception as ex:
            print("Unable to train the model - ", ex)

        return trainer

    def save_artifacts(model, tokenizer, model_name, tokenizer_name):
        """Save artifacts for the model predictions"""
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)

    def save_model(self, model, tokenizer, loc_name, label_list):
        """saves the artificats to given location"""
        model.save_pretrained(loc_name)
        tokenizer.save_pretrained("tokenizer")
        print("Successfully saved the model :)")

