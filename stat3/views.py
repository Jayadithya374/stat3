import joblib
import os
from django.shortcuts import render
from .forms import SMILESFORM
from rdkit import Chem
from rdkit.Chem import Descriptors
from keras.models import load_model
from sklearn import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def home(request):

    inputForm = SMILESFORM()

    if request.method == 'POST':

        inputForm = SMILESFORM(request.POST)

        if inputForm.is_valid():

            SMILES = request.POST.get('SMILES')
            prediction = Prediction(SMILES)

            if prediction:
                prediction = 'This molecule can be a STAT3 inhibitor.'
            else:
                prediction = 'This molecule may not be a STAT3 inhibitor.'
            
            return render(request, 'home.html', {'form': inputForm, 'prediction': prediction})
        
        else:

            return render(request, 'home.html', {'form': inputForm, 'error':inputForm.errors})
    
    return render(request, 'home.html', {'form': inputForm})



def Prediction(SMILES: str):

    molecule = Chem.MolFromSmiles(SMILES)
    descriptors = []
    noUseColumns = ['NumRadicalElectrons','SMR_VSA8','SlogP_VSA9',
                    'fr_SH','fr_azide','fr_benzodiazepine','fr_diazo',
                    'fr_epoxide','fr_isocyan','fr_isothiocyan','fr_phos_acid',
                    'fr_phos_ester','fr_prisulfonamd','fr_quatN','fr_thiocyan'
                    ]
    for rdkitDescriptorName, rdkitDescriptorFunction in Descriptors.descList:
        if rdkitDescriptorName not in noUseColumns:
            descriptors.append(rdkitDescriptorFunction(molecule))
    
    minmaxscaler = joblib.load('stat3/implementedModels/minmaxscaler.gz')
    descriptors = minmaxscaler.transform([descriptors])

    standardscaler = joblib.load('stat3/implementedModels/standardscaler.gz')
    descriptors = standardscaler.transform(descriptors)

    nnmodel = load_model('stat3/implementedModels/neuralNetwork.h5')
    nnprediction = nnmodel.predict([descriptors]).flatten()[0]

    logisticmodel = joblib.load('stat3/implementedModels/logistic.pkl')
    logisticprediction = logisticmodel.predict(descriptors).flatten()[0]

    svmmodel = joblib.load('stat3/implementedModels/supportVector.pkl')
    svmprediction = svmmodel.predict(descriptors).flatten()[0]

    if nnprediction+logisticprediction+svmprediction > 2.5:
        prediction = True
    else:
        prediction = False

    return prediction