@echo off

pip install uv
uv --version

uv venv --python 3.11
uv venv --path .venv

call .venv\Scripts\activate.bat

uv pip install -r requirements.txt

python main.py

pause
