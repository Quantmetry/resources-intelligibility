# resources-intelligibility

Some resources for intelligibility analysis of machine learning models, mostly in French.

Notes on setting up the project
-------------------------------

- with a version of *python3* installed (tested with python 3.6), make sure you have access to *pip*.
- with the below instructions, create a local virtual environnment and activate it
- install requirements.txt

  ```
  $ python3 -m venv .venv
  $ source .venv/bin/activate
  (.venv) $ pip install -r requirements.txt
  ```

- Go to the *data/* folder and download (~0.5Mo) the required data with the link you can find in *data/howtogetdata.txt*. At the end of this step, you should have a *carInsurance_train.csv* file in the *data/* folder.
- Start a jupyter server.

  ```
  (.venv) $ jupyter notebook
  ```

Features
--------

In the *notebooks/* folder, you will find some demos of several intelligibility techniques:

- Partie1\_Construction\_Modèle.ipynb
- Partie2\_Analyse_sensibilité\_des\_prédictions.ipynb
- Partie3\_Décomposition\_en\_contributions.ipynb
- Partie4\_Décomposition\_en\_règles.ipynb

You should run *Partie1* first because it will write a pickle with data and model, used by other notebooks. Afterwards, notebooks are independant.


Credits
-------
This work has been done by Quantmetry R&D, 2018.