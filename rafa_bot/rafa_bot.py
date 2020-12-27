#autor Rafael RR
#materia Inteligencia artificial 2021 - 1
import pandas as pd
import numpy as np
from sklearn import tree
import pickle
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

archivo_train="base.csv"
nombre_ia_tema="rafa_bot_tema.pkl"
nombre_ia_tipo="rafa_bot_tipo.pkl"


def procesa_texto( invisible = 0):
    cols=["tema","sentimiento","texto","tipo"]
    cols_train=["texto"]
    col_layer=["tema"]
    df=pd.read_csv(archivo_train)    
    X_df = df.loc[:, cols]
    X_df =sklearn.utils.shuffle(X_df)
    
    text = X_df[cols_train]
    Y_tema = X_df["tema"]
    Y_tipo = X_df["tipo"]
    
    l_texto=[]
    
    for i in range(0, len(text)):
        # Remove all the special characters
        texto_aux = re.sub(r'\W', ' ', str(text["texto"][i]))
        
        # Substituting multiple spaces with single space
        texto_aux = re.sub(r'\s+', ' ', texto_aux, flags=re.I)
        #print(text["texto"][i])
        
        # Converting to Lowercase
        texto_aux = texto_aux.lower()
        
        # remover stopwords
        
        l_texto.append(texto_aux)
    
    #bag of words
    #puedes crear una bolsa de palabras de todas las palabras en espaniol
    vectorizer = CountVectorizer(stop_words=stopwords.words('spanish'))
    X = vectorizer.fit_transform(l_texto).toarray()
    
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_tema, test_size=0.2, random_state=10)
    modelo_tema=crea_modelo(data=X_train, etiqueta=y_train)
    
    X_train_tipo, X_test_tipo, y_train_tipo, y_test_tipo = train_test_split(X, Y_tipo, test_size=0.2, random_state=7)
    modelo_tipo=crea_modelo(data=X_train_tipo, etiqueta=y_train_tipo)
    
    if invisible!=0:
        estadistica_modelo(modelo_tema, X_test, y_test, nombre_ia_tema)
        estadistica_modelo(modelo_tipo, X_test_tipo, y_test_tipo, nombre_ia_tipo)
    return [vectorizer, tfidfconverter]
    
def estadistica_modelo(modelo, X_test, y_test, nombre_ia):
    y_pred = modelo.predict(X_test)
    
    #Evaluating the Model
    print ("---------------------------------------\n")
    print ("matriz de confunsion")
    print(confusion_matrix(y_test,y_pred))
    
    print ("---------------------------------------\n")
    print(classification_report(y_test,y_pred))
    
    print ("---------------------------------------\n")
    exito=int(accuracy_score(y_test, y_pred)*100)
    print("el exito del modelo es: "+str(exito)+ "%")
    
    print ("---------------------------------------\n")
    #guardamos el modelo 
    with open(nombre_ia, 'wb') as picklefile:
        pickle.dump(modelo,picklefile)


def usa_modelo(vectorizer, tfidfconverter, texto, nombre_ia):
    #docs_new = ['hola soy nuevo', 'jonh titor']
    docs_new = [texto]
    X_new_counts = vectorizer.transform(docs_new)
    X_new_tfidf = tfidfconverter.transform(X_new_counts).toarray()
    
    with open(nombre_ia, 'rb') as training_model:
        modelo = pickle.load(training_model)
    
    predicted = modelo.predict(X_new_tfidf)
    return predicted
    
def crea_modelo(data, etiqueta):
    ##se crea el arbol
    #modelo= tree.DecisionTreeClassifier(criterion='entropy'          #gini, entropy
                                       #,splitter='best'             #best,random
                                       #,max_depth=None              #5,None
                                       #,min_samples_split=2         # int, float, optional (default=2)
                                       #,min_samples_leaf=1          # int, float, optional (default=1)
                                       #,min_weight_fraction_leaf=0.0001    # float, optional (default=0.)
                                       #,max_features=None        # int, float, string or None, optional (default=None ) ,auto, log, sqrt, 
                                       #,random_state=None                #None,int
                                       #)
    
    #modelo= RandomForestClassifier(n_estimators=1000, random_state=0)
    modelo = BernoulliNB()
    modelo.fit(data, etiqueta)
    return modelo


def get_saludo():
    cols=["tema","texto"]
    df=pd.read_csv(archivo_train)    
    X_df = df.loc[:, cols]
    X_df =sklearn.utils.shuffle(X_df)
    saludo_ran=(X_df[X_df.tema=="saludo"].sample()["texto"].iloc[0])
    return saludo_ran

def get_random_comment(tema, tipo):
    cols=["tema","texto","tipo"]
    saludo_ran=''
    df=pd.read_csv(archivo_train)    
    X_df = df.loc[:, cols]
    X_df =sklearn.utils.shuffle(X_df)
    
    aux_ran=X_df[X_df.tema==tema]
    #print (aux_ran.head())
    if tipo =="comentario":
        saludo_ran=(aux_ran[aux_ran.tipo=="pregunta"].sample())
    else:
        saludo_ran=(aux_ran[aux_ran.tipo=="comentario"].sample())
        
    return saludo_ran

def get_random_comment_otro_tema(tema, tipo):
    cols=["tema","texto","tipo"]
    saludo_ran=''
    df=pd.read_csv(archivo_train)    
    X_df = df.loc[:, cols]
    X_df =sklearn.utils.shuffle(X_df)
    
    aux_ran=X_df[X_df.tema==tema]
    #print (aux_ran.head())
    if tipo =="pregunta":
        saludo_ran=(aux_ran[aux_ran.tipo=="pregunta"].sample())
    else:
        saludo_ran=(aux_ran[aux_ran.tipo=="comentario"].sample())
        
    return saludo_ran


def usa_modelo_tema(vectorizer, tfidfconverter, docs_new):
    return usa_modelo(vectorizer, tfidfconverter, docs_new, nombre_ia_tema)

def usa_modelo_tipo(vectorizer, tfidfconverter, docs_new):
    return usa_modelo(vectorizer, tfidfconverter, docs_new, nombre_ia_tipo)

#l_res=procesa_texto( invisible = 1)
#vectorizer=l_res[0]
#tfidfconverter=l_res[1]
#docs_new = 'hola soy nuevo'
#print(usa_modelo(vectorizer, tfidfconverter, docs_new, nombre_ia_tema))
