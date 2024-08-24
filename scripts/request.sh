#!/bin/bash
curl --request POST \
    --url http://localhost:8000/v1/completions \
    --header "Content-Type: application/json" \
    --data "{\"prompt\": \"write just what i say, do not say anything else: make me a bash script to remove all files from a directory:\",\"n_predict\": 128,
    \"temperature\":0, \"logprobs\":true, \"stop\":[\"Q:\", \"\n\"]}"


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
