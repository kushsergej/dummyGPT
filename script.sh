#!/bin/bash

pip install uv
uv --version

uv venv --python 3.13 --path .venv

# Windows Git Bash compatible activation
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment activation script not found"
    exit 1
fi

uv add -r requirements.txt

uv run main.py