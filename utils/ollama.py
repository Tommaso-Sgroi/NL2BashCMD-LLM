import time
from typing import Literal, List, Dict

import ollama


def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        return te - ts, result
    return timed

num = 0
def call_ollama(model, messages: List[Dict], *, options):
    response = ollama.chat(model=model, messages=messages, stream=True, options=options)
    tokens = []
    global num
    for r in response:
        tokens.append({

            {
                'token': r['message']['content'],
                'probability': r['completion_probabilities']
            }
        }
        )
    num += 1
    return response
