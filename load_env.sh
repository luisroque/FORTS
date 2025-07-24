#!/bin/bash

# Add the project root to the PYTHONPATH to make the 'forts' module importable
export PYTHONPATH=".:${PYTHONPATH}"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
else
  echo ".env file not found. Please create one from .env.example."
fi
