#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# This part ensures the environment is clean for Streamlit