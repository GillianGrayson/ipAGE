# ipAGE
Predictor of age based on inflammatory/immunological profile


## Description
Performs statistical analysis and builds ipAGE clock according to the manuscript: https://www.biorxiv.org/content/10.1101/2021.07.23.453588v2

## How to run
Install dependencies:
```yaml
# clone project
git clone https://https://github.com/GillianGrayson/ipAGE
cd ipAGE

# install requirements in the current python environment
pip install -r requirements.txt
```
Run statistical analysis:
```yaml
python statistics.py 
```

Create the ipAGE clock:
```yaml
python create_clock.py 
```
<br>