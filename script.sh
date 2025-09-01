#!bin/bash

pip install uv
uv --version

uv venv --python 3.11
uv venv --path .venv
source .venv/bin/activate

uv add -r requirements.txt

uv run main.py