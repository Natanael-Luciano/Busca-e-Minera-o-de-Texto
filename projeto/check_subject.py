from joblib import dump, load

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import json
from joblib import dump, load


with open('mail_dict.json', 'r') as arquivo:
    mail_dict = json.load(arquivo)


with open('categorias_dict.json', 'r') as arquivo:
    categorias_dict = json.load(arquivo)

def cat_original(cat):
    if cat == 2:
        return 4
    if cat == 3:
        return 5
    if cat == 4:
        return 7
    return cat

def my_prepro(email):
        email_text = str(email)
        email_text = re.sub(r'[.,:;()\[\]-]', '', email_text)  # Remove os caracteres especificados
        email_text = re.sub(r'\n', ' ', email_text)  # Substitui quebras de linha por espaços
        email_text = re.sub(r'"', '', email_text)  # Remove aspas duplas
        email_text = email_text.lower()  # Converte para minúsculas

        return email_text
    
def preprocess_email(text):
        nltk.download('stopwords')

        # Definir o tokenizer que remove números e pontuações
        tokenizer = RegexpTokenizer(r'\w+')

        # Carregar stop words em inglês
        stop_words = set(stopwords.words('portuguese'))
        # Tokeniza o texto removendo pontuações e números
        tokens = tokenizer.tokenize(text.lower())
        
        # Remove as stop words
        filtered_words = [word for word in tokens if word not in stop_words and word.isalpha()]
        
        return filtered_words


def mail_to_vec(email, mail_dict : list):

        # Cria um dicionário para contar as ocorrências
        word_count = {word: 0 for word in mail_dict}

        # Conta as ocorrências das palavras do texto que estão no dicionário
        for word in email:
            if word in word_count:
                word_count[word] += 1

        # Cria o vetor de ocorrências com base na ordem das palavras do dicionário

        vector = np.array([word_count[word] for word in mail_dict])

        return vector/(np.linalg.norm(vector) + 0.001)
 
def check_sub(email, model_path, pca_path):
        model = load(model_path)
        pca = load(pca_path)

        email = my_prepro(email)
        token_mail = preprocess_email(email)
        vec_mail = mail_to_vec(token_mail, mail_dict).reshape(1,-1)
        vec_mail = pca.transform(vec_mail)
        cat_t = model.predict(vec_mail)
        return str(cat_original(cat_t))
 
if __name__ == '__main__':
    test_email = """Fala Bob. Tudo bem?
                    Tava querendo dar entrada com meu processo de equivalência de disciplina
                    Fiz a matéria XXXXXXX e queria pegar equivalencia em YYYYYY.
                    Meu DRE é **********

                    Att: ciclando deca"""


    cat = check_sub(test_email, 'SVC_T1.joblib', 'pca.joblib')
    print(cat)
    print(f"O email foi categorizado como {categorias_dict[cat]}")