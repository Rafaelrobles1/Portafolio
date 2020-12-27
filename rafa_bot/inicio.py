#autor Rafael RR
#Materia Inteligencia artificial 2021 - 1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import pickle
import re
import nltk
import os


import rafa_bot
 
def menu():
    """
    Funcion que limpia la pantalla y muestra nuevamente el menu
    """
    print ("Selecciona una opcion")
    print ("\t1 - Entrenar al modelo")
    print ("\t2 - Modo test ..")
    print ("\t3 - Modo normal, se termina escribiendo 'adios' ..")
    print ("\t9 - Salir")
 
 
os.system('cls') # NOTA para windows tienes que cambiar clear por cls
l_contexto=[]
while True:
    # Mostramos el menu
    menu()
    modo_infinito=True
    # solicituamos una opcion al usuario
    opcionMenu = input("inserta un numero valor >> ")
    l_contexto=rafa_bot.procesa_texto(invisible=0)
    
    if opcionMenu==1:
        print ("\nEntrenando modelo .. \n")
        l_contexto=rafa_bot.procesa_texto(invisible=1)
        print ("si deseas un mejor resultado entrena al modelo de nuevo.\n")
        
    elif opcionMenu==2:
        print ("\nrafa_bot  >> "+  str(rafa_bot.get_saludo())+ "\n")
        usuario_res = raw_input('respuesta >> ')

        #determina el bot del tema conversacion y da un comentario al azar de ese mismo tema
        vectorizer=l_contexto[0]
        tfidfconverter=l_contexto[1]
        texto=usuario_res
        
        tema=rafa_bot.usa_modelo_tema(vectorizer, tfidfconverter, texto)
        tipo=rafa_bot.usa_modelo_tipo(vectorizer, tfidfconverter, texto)
        
        print ("\nrafa_bot determino el tema : "+str(tema[0])+"\n")
        print ("\nrafa_bot determino el tipo : "+str(tipo[0]))
        coment_ran=rafa_bot.get_random_comment(tema[0],tipo[0])

        print ("--------------------------------------------------")
        print ("rafa_bot usa el tema            : "+str(coment_ran["tema"].iloc[0])+"\n" )
        print ("rafa_bot el tipo de respuesta es: "+str(coment_ran["tipo"].iloc[0])+"\n" )
        print ("rafa_bot dice                   : "+str(coment_ran["texto"].iloc[0])+"\n" )
    

    
    elif opcionMenu==3:
        print ("\nrafa_bot  >> "+  str(rafa_bot.get_saludo())+ "\n")
        usuario_res = raw_input('respuesta     >> ')
         
        while(modo_infinito):
            if str(usuario_res)!="adios":
                vectorizer=l_contexto[0]
                tfidfconverter=l_contexto[1]
                texto=usuario_res
            
                tema=rafa_bot.usa_modelo_tema(vectorizer, tfidfconverter, texto)
                tipo=rafa_bot.usa_modelo_tipo(vectorizer, tfidfconverter, texto)
                
                coment_ran=rafa_bot.get_random_comment(tema[0],tipo[0])
                print ("\nrafa_bot dice : "+str(coment_ran["texto"].iloc[0])+"\n" )
                
                usuario_res = raw_input('usuario dice >> ')
                
                #para evitar ciclar la conversacion
                tipo_new=rafa_bot.usa_modelo_tipo(vectorizer, tfidfconverter, usuario_res)
                if tipo== tipo_new:
                    coment_ran=rafa_bot.get_random_comment_otro_tema(tema[0],tipo[0])
                
                print ("\nrafa_bot dice : "+str(coment_ran["texto"].iloc[0])+"\n" )
                
                usuario_res = raw_input('usuario dice >> ')
                
            else:
                modo_infinito=False
            
        
    elif opcionMenu==9:
        break
    else:
        print ("")
        input("No has pulsado ninguna opcion correcta...\npulsa una tecla para continuar")
