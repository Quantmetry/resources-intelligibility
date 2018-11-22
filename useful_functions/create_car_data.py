"""car insurance data set

"""
# Author: Jean-Matthieu Schertzer <jmschertzer@quantmetry.com>


import pandas as pd
import numpy as np
from sklearn.datasets.base import get_data_home, Bunch
from sklearn.datasets.base import _fetch_remote, RemoteFileMetadata
from os.path import exists, join
from .mltask import MLTask
from sklearn.model_selection import train_test_split

def load_car_data():

    try:
        data = pd.read_csv('../data/carInsurance_train.csv'
            )
        print('Data were correctly read.')
    except:
        raise ValueError("DEBUG: Couldn't load car Insurance data.")

    # Remove rows with some variable missing
    data = data.query('Job == Job')
    data = data.query('Education == Education')

    dataset = Bunch(
        data=(data.drop('CarInsurance', axis=1)),
        target=np.array(data['CarInsurance'])
        )
    return dataset


def prepare_car_data(bunch_data):
    X = bunch_data.data
    y = bunch_data.target

    # Preparing target and variables
    data = pd.DataFrame(X, columns=X.columns)
    id_data = data['Id']

    for col in ['Id']:
        del data[col]

    data['Communication'].fillna("unknown", inplace=True)
    data['Outcome'].fillna("unknown", inplace=True)

    # Renaming a column more logically
    data = data.rename(columns={
        "Age": "AGE",
        "Job": "JOB",
        "Marital": "MARITAL",
        "Education": "EDUCATION",
        "Default": "DEFAUT",
        "Balance": "SOLDE",
        "HHInsurance": "ASSUR_HABITATION",
        "CarLoan": "PRET_AUTO",
        "Communication": "CANAL_COM",
        "LastContactMonth": "MOIS_CONTACT",
        "LastContactDay": "JOUR_CONTACT",
        "CallStart": "HEURE_DEBUT_CONTACT",
        "CallEnd": "HEURE_FIN_CONTACT",
        "NoOfContacts": "NB_CONTACT",
        "DaysPassed": "DUREE_DEPUIS_CONTACT",
        "PrevAttempts": "NB_ANCIEN_CONTACT",
        "Outcome": "RESULTAT_PRECEDENT"})



    # Restauring explicitely categorical features
    data = data.replace({
        "JOB": {
            "management": "manager",
            "blue-collar": "ouvrier",
            "technician": "technicien",
            "admin.": "administratif",
            "services": "administratif",
            "retired": "retraite",
            "self-employed": "autoentrepreneur",
            "unemployed": "sans_emploi",
            "entrepreneur": "autoentrepreneur",
            "student": u"étudiant",
            "housemaid": "au_foyer"
            },
        "MARITAL": {"single": u"célibataire",
                      "married":u"marié",
                      "divorced":u"divorcé"
                     },
        "EDUCATION": {"primary": 1.0,
                       "secondary": 2.0,
                       "tertiary": 3.0
                      },
        "CANAL_COM": {"cellular": "mobile",
                           "telephone": "fixe",
                           "unknown": "inconnu"
                           },
        "RESULTAT_PRECEDENT": {"failure": u"échec",
                                    "other": "autre",
                                    "success": u"succès",
                                    "unknown": "inconnu"}
    })


    data['HEURE_FIN_CONTACT'] = pd.to_datetime(data['HEURE_FIN_CONTACT'])
    data['HEURE_DEBUT_CONTACT'] = pd.to_datetime(data['HEURE_DEBUT_CONTACT'])
    data['DUREE_CONTACT'] = ((data['HEURE_FIN_CONTACT'] - data['HEURE_DEBUT_CONTACT'])/np.timedelta64(1,'m')).astype(float)
    data['DUREE_CONTACT'] = round(data['DUREE_CONTACT'], 1)

    for col in ['HEURE_FIN_CONTACT', 'HEURE_DEBUT_CONTACT', 'MOIS_CONTACT',
        'JOUR_CONTACT']:
        del data[col]


    # Creating metadata
    feature_names = list(data.columns)
    categorical_names = ["JOB", "MARITAL", "CANAL_COM", "RESULTAT_PRECEDENT"]
    numerical_names = [feat for feat in feature_names if feat not in categorical_names]
    NLP_column_ids = []
    task_type = "binary_classification"

    # Creating the train/test split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        data, y, id_data, random_state=42, test_size=0.5)

    # Creating MLTask
    task_car_data = MLTask()

    task_car_data.X_train = X_train
    task_car_data.X_test = X_test
    task_car_data.y_train = y_train
    task_car_data.y_test = y_test
    task_car_data.id_train = id_train
    task_car_data.id_test = id_test
    task_car_data.feature_names = feature_names
    task_car_data.categorical_names = categorical_names
    task_car_data.numerical_names = numerical_names
    task_car_data.NLP_column_ids = NLP_column_ids
    task_car_data.task_type = task_type

    return task_car_data
