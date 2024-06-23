import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Dicinario de categorias
categorias_dict = {
    0: "Outros",
    1: "IC (Projeto I e II)",
    2: "Formatura",
    3: "inscrição em disciplinas",
    4: "Equivalências/Isenções",
    5: "Trancar/Destrancar matricula",
    6: "Respostas/vazio/invalido",
    7: "Alteração/Inclusão/Exclusão de grau",
    8: "Trancamento de materias",
    9: "Interno",
    10: "Mudança de turma",
    11: "Transferência de curso",
    12: "Regularizar inscrição",
    13: "Documentos",
    14: "Pedido e retirada de diploma",
    15: "Pedido de email @matematica",
    16: "Alocamento de sala e vagas",
    17: "Inscrição de calouros",
    18: "Assinatura de estágio",
    19: "Erro do SIGA ou Sistema ou Secretário ou ACKER",
    20: "Quebra de requisitos",
    21: "Divulgação",
    22: "Requerimento ligado as PR",
    23: "Processos e requerimento(SEI)",
    24: "Encaminhento para Pós",
    25: "Tradução Juramento",
    26: "Declaração de Ênfase",
    27: "(Des)Cancelamento de Matrícula",
    28: "Jubilamento",
    29: "COAA",
    30: "Reingresso"
}

with open('categorias_dict.json', 'w') as arquivo:
    json.dump(categorias_dict, arquivo)

# Caminho do arquivo CSV
file_path = 'emails_aplicada_classificado.csv'

# Carregar os dados do CSV
df = pd.read_csv(file_path)

# Inicializar a lista para armazenar os resultados
resultados = []

# Ler a coluna 'categoria', processar cada entrada e adicionar à lista
for item in df['Categoria']:
    # Remove espaços em branco
    item = item.replace(' ', '')
    item = item.replace('.', ',')
    # Faz split por vírgula

    split_items = item.split(',')
    # Converte os resultados do split para inteiros e adiciona à lista
    resultados.extend([int(num) for num in split_items])

# Imprimir a lista de resultados
# print(resultados)


# Processar a coluna 'categoria'
df['Categoria'] = df['Categoria'].apply(lambda x: x.replace(' ', ''))
df['Categoria'] = df['Categoria'].apply(lambda x: x.replace('.', ','))
df['Categoria'] = df['Categoria'].apply(lambda x: [int(n.strip()) for n in x.split(',')])

# Criar a função para aplicar a condição


def processar_categorias(categorias):
    relevantes = [4, 5, 7, 1]
    contidos = [cat for cat in categorias if cat in relevantes]

    # Categorização efetiva
    if len(contidos) == 0:
        return [0]
    elif len(contidos) == 1:
        return contidos
    else:
        return contidos

# Criar nova coluna 'categoria_efetiva' com processamento
df['categoria_efetiva'] = df['Categoria'].apply(processar_categorias)

# Expandir o DataFrame quando houver múltiplas categorias efetivas
rows = []
for _, row in df.iterrows():
    for cat in row['categoria_efetiva']:
        new_row = row.copy()
        new_row['categoria_efetiva'] = cat
        rows.append(new_row)

# Criar novo DataFrame expandido
new_df = pd.DataFrame(rows)

# Reinicializar índice no novo DataFrame
new_df.reset_index(drop=True, inplace=True)

# Mostrar o novo DataFrame
print(new_df)

if __name__ == '__main__':

    contagem = Counter(resultados)

    # Separar os elementos e suas contagens
    elementos = list(contagem.keys())
    ocorrencias = list(contagem.values())

    # Mapear os elementos numéricos para os nomes usando o dicionário
    elementos_nomes = [categorias_dict[elem] if elem in categorias_dict else "Desconhecido" for elem in elementos]

    # Criar um gráfico de barras
    plt.figure(figsize=(15, 8))  # Ajuste no tamanho para melhor visualização
    plt.bar(elementos_nomes, ocorrencias, color='blue')

    # Adicionar título e rótulos
    plt.title('Contagem de Ocorrências dos Elementos')
    plt.xlabel('Categorias')
    plt.ylabel('Ocorrências')

    # Rotação dos labels no eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')  # Ajustar a alinhamento para 'right'

    # Ajustar o layout para evitar cortes
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()

    # ######################################################################################################

    # Contagem das ocorrências na coluna 'categoria_efetiva'
    contagem = new_df['categoria_efetiva'].value_counts()

    # Ordenar o índice da série de contagem (que são as categorias efetivas)
    contagem_sorted = contagem.sort_index()

    # Substituir os índices numéricos pelos nomes das categorias usando o dicionário

    contagem_sorted.index = [categorias_dict[idx] for idx in contagem_sorted.index]

    # Criar o gráfico de barras
    plt.figure(figsize=(15, 8))  # Tamanho do gráfico ajustado para melhorar a visualização
    contagem_sorted.plot(kind='bar', color='blue')  # Gráfico de barras

    # Adicionar título e rótulos
    plt.title('Contagem de Ocorrências por Categoria Efetiva')
    plt.xlabel('Categoria Efetiva')
    plt.ylabel('Número de Ocorrências')
    # Rotação dos labels no eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')  # Ajustar a alinhamento para 'right'

    # Ajustar o layout para evitar cortes
    plt.tight_layout()
    # Mostrar o gráfico
    plt.show()
    print("aaaa")

    # ######################################################################################################

    print(new_df.head())
    print('\n')
    print(np.sum(new_df['categoria_efetiva'].value_counts()))

    # ######################################################################################################

    def my_prepro(email):
        email_text = str(email)
        email_text = re.sub(r'[.,:;()\[\]-]', '', email_text)  # Remove os caracteres especificados
        email_text = re.sub(r'\n', ' ', email_text)  # Substitui quebras de linha por espaços
        email_text = re.sub(r'"', '', email_text)  # Remove aspas duplas
        email_text = email_text.lower()  # Converte para minúsculas

        return email_text
    
    new_df['Texto'] = new_df['Texto'].apply(my_prepro)

    # Baixar a lista de stop words
    nltk.download('stopwords')

    # Definir o tokenizer que remove números e pontuações
    tokenizer = RegexpTokenizer(r'\w+')

    # Carregar stop words em inglês
    stop_words = set(stopwords.words('portuguese'))

    def preprocess_email(text):
        # Tokeniza o texto removendo pontuações e números
        tokens = tokenizer.tokenize(text.lower())
        
        # Remove as stop words
        filtered_words = [word for word in tokens if word not in stop_words and word.isalpha()]
        
        return filtered_words


    def build_dictionary(df, column_name):
        word_count = {}

        # Contar as ocorrências de cada palavra
        for email in df[column_name]:
            processed_words = preprocess_email(email)
            for word in processed_words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        # Filtrar palavras que aparecem mais de 5 vezes
        frequent_words = [word for word, count in word_count.items() if count > 5]

        return frequent_words

    # Construir o dicionário
    mail_dict = build_dictionary(new_df, 'Texto')
    # Salvando a lista em um arquivo JSON
    with open('mail_dict.json', 'w') as arquivo:
        json.dump(mail_dict, arquivo)
    print(mail_dict)

    new_df['Assunto'] = new_df['Assunto'].apply(my_prepro)

    print(new_df['Assunto'])

    new_df['Texto'] = new_df['Texto'].apply(preprocess_email)
    new_df['Assunto'] = new_df['Assunto'].apply(preprocess_email)

    print(new_df.head())

    # ######################################################################################################

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

    # ######################################################################################################

    new_df['Texto'] = new_df['Texto'].apply(mail_to_vec, args=(mail_dict,))

    print(new_df.head())

    new_df[['Assunto','Texto','Categoria','categoria_efetiva']].to_csv('vec_colação.csv', encoding='utf-8', index=False)

    new_df['Texto'].to_hdf('vec_colação.h5', key='vetores', mode='w')


print(len(mail_dict))