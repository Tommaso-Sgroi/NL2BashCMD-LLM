from json import JSONEncoder
from typing import Dict
from .utils import timeit
import requests
import json

class RequestData(JSONEncoder):
    def __init__(self, prompt, *, n_predict=0, temperature, logprobs, **kwargs):
        super().__init__()
        self.prompt = prompt
        self.n_predict =  n_predict
        self. temperature =  temperature
        self.logprobs = logprobs

        for k,v in kwargs.items():
            setattr(self, k, v)

    def toJson(self):
        return self.__dict__


class ResponseData(object):
    def __init__(self, data):
        self.model = data["model"]
        self.text = data["choices"][0]["text"]
        self.logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
        self.tokens = data["choices"][0]["logprobs"]["tokens"]

    def remove_backticks(self):
        if '`' in self.tokens[0]:
            del self.tokens[0]
            del self.logprobs[0]
            self.text.replace('`', '', 1)

        if '`' in self.tokens[-1]:
            del self.logprobs[-1]
            del self.logprobs[-1]

        if '\n' in self.tokens:
            del self.tokens[-1]
            del self.logprobs[-1]


    def clean_text(self):
        self.text = self.text.strip()

    def __str__(self):
        return f'[model]{self.model}: {self.text}'


@timeit
def chat_llama_cpp_server(*, url, headers, data:RequestData):
    response = requests.post(url, headers=headers, data=json.dumps(data.toJson()))
    json_response = response.json()
    rd = ResponseData(json_response)
    return rd
