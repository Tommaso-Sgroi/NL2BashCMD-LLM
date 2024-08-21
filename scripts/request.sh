#!/bin/bash
curl --request POST \
    --url http://localhost:8000/v1/completions \
    --header "Content-Type: application/json" \
--data '{"prompts": [" What is the capital of the US?", " What is the capital of France? "], "n_predict": 2048}'


#curl --request POST \
#    --url http://localhost:8000/v1/chat/completions \
#    --header "Content-Type: application/json" \
#    --data '{
#        "messages": [
#            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#            {"role": "user", "content": "Who are you?"}
#        ],
#        "n_predict": 128,
#        "temperature": 0,
#        "logprobs": true,
#        "stop": ["Q:", "\n"],
#        "stream": true
#    }'

printf "\n"
