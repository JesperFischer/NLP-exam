python3 -m venv env

source ./env/bin/activate
sudo apt-get install python3-dev
pip install --upgrade pip
pip install -r requirements.txt

python3 scripts/Main_analysis.py

deactivate
