import zipfile
import os

# Descompacta o dataset baixado
with zipfile.ZipFile("cassava-leaf-disease-classification.zip", 'r') as zip_ref:
    zip_ref.extractall("cassava_data")