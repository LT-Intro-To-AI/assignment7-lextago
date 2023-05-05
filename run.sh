# if run.sh is present, it will be what runs when you press "Run"

# -q installs quietly
echo 'Installing things please be patient!'
pip install -q -r requirements.txt
clear

# then we have to manually run
python proj.py