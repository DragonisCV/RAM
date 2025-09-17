### exit on error
set -e
### set conda hook
eval "$(conda shell.bash hook)"

### create enviroment
conda create -n RAM2 -y python=3.12
conda activate RAM2

### install package
pip install -r requirements.txt
python setup.py develop
