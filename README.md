# To run

First set up and activate a virtual environment and install the dependencies. You
may need to use `python3` instead of `python` on your system.
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
To replicate each figure run the corresponding script generator, then the resulting
script, and then the plotting scripts. After activating the virtual environment
one should always use `python` rather than `python3`.
```
python fig(x)scriptgenerator.py
python fig(x)script.py
python fig(x)repro.py
```
