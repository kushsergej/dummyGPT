#!/bin/bash

python -m pip install --upgrade uv
uv venv .venv --python 3.12 --clear
source .venv/Scripts/activate

uv add torch dotenv

uv run main.py