import os
from json import JSONEncoder
from .utils import timeit
import requests
import json
from time import sleep as s

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
    def __init__(self, data, i=0):
        self.model = data["model"]
        self.text = data["choices"][i]["text"]
        self.logprobs = data["choices"][i]["logprobs"]["token_logprobs"]
        self.tokens = data["choices"][i]["logprobs"]["tokens"]

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
class ResponseChatData(ResponseData):
    def __init__(self, data, i=0):
        self.model = data["model"]
        self.text = data["choices"][i]["message"]["content"]
        self.text = self.text.replace('```bash\n', '').replace('```', '')
        self.logprobs = []
        self.tokens = []
        for logprobs in data["choices"][i]["logprobs"]["content"]:
            lp = logprobs['logprob']
            tk = logprobs['token']
            self.logprobs.append(lp)
            self.tokens.append(tk)


def wait_rate_limit(rate):
    s(rate + 0.01)

@timeit
def chat_llama_cpp_server(*, url, headers, data:RequestData):
    response = requests.post(url, headers=headers, data=json.dumps(data.toJson()))
    json_response = response.json()
    rd = ResponseData(json_response)
    return rd

def inference_with_together_api(*, url, headers, data, use_proxy=False):
    response = requests.post(url, data=json.dumps(data), headers=headers, \
                             proxies=None if not use_proxy else {'https':os.getenv('PROXY_PIA')})
    json_response = response.json()
    if response.status_code != 200:
        print("Error: " + json.dumps(json_response, indent=2))
    rd = ResponseData(json_response)
    return rd


def inference_with_api(*, url, headers, data, proxies=None):
    response = requests.post(url, data=json.dumps(data), headers=headers, proxies=proxies)
    json_response = response.json()
    if response.status_code != 200:
        print("Error: " + json.dumps(json_response, indent=2))
    rds = []
    for i in range(len(json_response['choices'])):
        rd = ResponseData(json_response, i)
        rds.append(rd)
    return rds