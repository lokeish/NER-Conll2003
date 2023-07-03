from fastapi import FastAPI, HTTPException
from base_models import Input
from helper import format_result
import logging
import model

app = FastAPI()
model_obj = model.Model()

@app.post('/predict')
async def predict(input: Input):
    response = model_obj.predict(input.message)
    clear_resp = format_result(response)
    output = {"clear_response": clear_resp, "raw_response": str(response)}
    return output

