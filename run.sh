

python3 -m venv env 
source ./env/bin/activate 
sudo apt-get update 
sudo apt-get install python3-dev
sudo apt-get install python3-venv 
pip install --upgrade pip 
pip install -r requirements.txt 
python3 scripts/Main_analysis.py 

deactivate