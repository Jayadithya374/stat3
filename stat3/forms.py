from django import forms

class SMILESFORM(forms.Form):

    SMILES = forms.CharField(label='SMILES', max_length=1000, required=True,)
