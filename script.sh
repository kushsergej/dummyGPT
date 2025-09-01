#!/bin/bash

pip3 venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt

python3 main.py