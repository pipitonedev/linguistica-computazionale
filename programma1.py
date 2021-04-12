# -*- coding: utf-8 -*-

import sys
import codecs
import math
import nltk
from nltk import bigrams
from math import log


#funzione che divide le frasi in Token

def CalcolaToken(frasi):
    tokensTOT=[]
    for frase in frasi:
        tokens=nltk.word_tokenize(frase)                        #divido la frase in token
        tokensTOT=tokensTOT+tokens                              #creo la lista che contiene tutti i token del testo
    return tokensTOT                                            #restituisco i risultati

#Funzione che calcola la quantità di caratteri nel testo

def NumeroCaratteri(corpus): 
    count=0
    for tokens in corpus:                                    #ciclo che considera ogni parola del corpus
        count = count + len(tokens)                          #viene sommata la lunghezza delle parole
    return count                                              #restituisce il risultato

#funzione che incrementa gli Hapax ogni 1000 Token

def HapaxIncremento(tokenTOT):
    ListaHapax = []                               #Lista che conterrà gli Hapax
    ListaIncrementale = []                        #Lista vuota che serve a concatenare le liste ogni 1000 token letti
    ListaVoc = []                                  #Lista con solo il vocabolario della lista incrementale
    ListaComposta = [tokenTOT[x:x+1000] for x in range(0, len(tokenTOT),1000)]            #viene creata una sottolista ogni 1000 Token letti
    for lista in ListaComposta:                                                           #Avviene il ciclo sulla lista composta per ogni sottolista
                     ListaIncrementale = ListaIncrementale + lista                        #Concateno la Lista composta con la lista Incremento
                     ListaVoc = set(ListaIncrementale)                                    #calcolo il vocabolario ogni volta che aggiungo 1000 token alla lista
                     NHapax= ContoHapax(ListaIncrementale,ListaVoc)                       #richiamo la funzione per il conteggio degli Hapax
                     ListaHapax.append(NHapax)                                             #appendo il numero trovato nella posizione i-esima nel vettore finale vocabolario
    return ListaHapax                                                                      #restituisco il risultato

#Funzione che conteggia gli Hapax

def ContoHapax(tokenTOT,vocabolario):
    conta = 0
    for tok in vocabolario:
        FreqToken = tokenTOT.count(tok)                                 #Calcolo la frequenza del token nel Corpus
        if FreqToken == 1:                                              #Se la frequenza del Token è 1 si tratta di un Hapax
            conta = conta + 1                                           #Conto quanti Hapax Trovo
    return conta

#Funzione che calcola il rapporto tra sostantivi e verbi

def RapportoSostantiviVerbi(POSTag):
    TagSostantivi = ["NN", "NNS", "NNP", "NNPS"]                  #Lista dei tag utilizzati nel POS Tagging (Penn Treebank)
    TagVerbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    ListaSostantivi = []                                             #Per ogni coppia memorizzo in due liste tutte le pos di sostantivi e verbi
    ListaVerbi = []
    for(tok,pos) in POSTag:
        if pos in TagSostantivi:
            ListaSostantivi.append(pos)
        if pos in TagVerbi:
            ListaVerbi.append(pos)
    OccorrenzaSostantivi = nltk.FreqDist(ListaSostantivi)               #Conto quante volte occorre ciascuna POS all'interno del Corpus
    OccorrenzaVerbi = nltk.FreqDist(ListaVerbi)
    sostantivi=sum(OccorrenzaSostantivi.values())                       #Faccio la somma di tutti i conteggi
    verbi=sum(OccorrenzaVerbi.values())
    return sostantivi*1.0/verbi*1.0                                      #Avviene il calcolo e restituisco il risultato finale

#funzione che estrae i POS più frequenti in tutto il testo con la relativa frequenza

def POSFrequenti(testoPOS): 
    listaTag = []
    for(tok,pos) in testoPOS:                                       #viene scansionato ogni token e il relativo POS
        listaTag.append(pos)                                        #viene aggiunto ogni token e il POS alla lista
        frequenza = nltk.FreqDist(listaTag)                         #conta quante volte occorre ciascun elemento della lista
        frequenza = frequenza.most_common(10)                       #vengono presi in considerazione soltanto n elementi più frequenti
    return frequenza

#funzione che viene invocata nel momento in cui bisogna ordinare una lista

def ordina(Lista):
    return sorted(Lista, reverse = True)

#Funzione che calcola la Probabilità Condizionata

def Condizionata(tokenPOS, ListaBigrammi):
    ListaBPCOND = []
    N = len(tokenPOS)
    DistrFreq = nltk.FreqDist(ListaBigrammi)                                    #Distribuzione di Frequenza all'interno della lista
    for bigramma in DistrFreq:                                                  #Inizio il ciclo della Probabilità Condizionata sui bigrammi di POS
        FreqA = tokenPOS.count(bigramma[0])
        FreqB = tokenPOS.count(bigramma[1])
        FreqAttesa = CalcoloFreqAttesa(FreqA,FreqB, N)                          #invoco la funzione che mi calcola la frequenza Attesa
        PCOND = (FreqAttesa/N*1.0)*100                                          #Calcolo la Probabilità Condizionata
        ListaBPCOND.append([PCOND, bigramma])                                   #Appendo alla lista i bigrammi con la rispettiva Prbabilità Condizionata
    ListaBPCOND = ordina(ListaBPCOND)                                           #ordino la lista in ordina decrescente in base alla probabilità, invocando la funzione di ordinamento
    return ListaBPCOND[:10]                                                     #Restituisco i risultati

#Funzione che calcola la Frequenza Attesa

def CalcoloFreqAttesa(a,b, N):
    totale = ((a*1.0)*(b*1.0))/(N*1.0)
    return totale

#Funzione che estrae soltanto i POS

def EstraiPOS(Lista):
    ListaFinale = []
    for elemento in Lista:
        ListaFinale.append(elemento[1])
    return ListaFinale

#Funzione che Calcolata la Forza Associativa massima dei bigrammi in termini di Local Mutual Information

def CalcoloLMI(tokenPOS, ListaBigrammi):
    ListaBigrammi = bigrams(tokenPOS)                                   #Lista che contiene i Bigrammi Pos
    Dizionario = {}                                                     #Dìzionario
    ListaFinale = []                                                    #Lista che restituirà il risultato Finale
    LMI = 0.0
    N = len(tokenPOS)
    DistrFreq = nltk.FreqDist(ListaBigrammi)                             #Distribuzione di frequenza all'interno della lista
    for bigramma in DistrFreq:                                           #Inizio il ciclo per il calcolo della LMI
        A = tokenPOS.count(bigramma[0]) 
        B = tokenPOS.count(bigramma[1])
        FO = DistrFreq[bigramma]                                        #Frequenza Osservata: <a,b>
        FA = CalcoloFreqAttesa(A,B,N)                                    #Invoco la funzione che mi calcola la frequenza attesa
        LMI = (FO*1.0)*math.log((FO*1.0)/(FA*1.0), 2)                   #LMI = F(a,b) * log2 ( F(a,b) / (F(a)*F(b)/N)) = frequenza osservata * log2 (frequenza osservata / frequenza attesa)
        ListaFinale.append([LMI, bigramma])                             #Appendo alla lista finale la local mutual information e i bigrammi
    ListaFinale = ordina(ListaFinale)                                   #ordino la lista in ordine decrescente in base alla Forza Associativa Massima, invocando la funzione di ordinamento
    return ListaFinale[:10]                                             #restiuisco i risultati
    
#FUNZIONE PRINCIPALE

def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8") #prende il file codificato in utf-8
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read() #viene letto il file e viene assegnato tutto il suo contenuto ad una variabile di tipo string
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #viene caricato il modello statistico 'english.pickle' su cui ci si basa per dividere il testo in frasi.
    frasi1 = sent_tokenizer.tokenize(raw1) ##Divido le frasi in token
    frasi2 = sent_tokenizer.tokenize(raw2)
    Corpus1 = CalcolaToken(frasi1) #Calcolo i token su tutto il corpus
    Corpus2 = CalcolaToken(frasi2)
    tokensPOS = nltk.pos_tag(Corpus1) #applica ad ogni token il tag di una parte del discorso per il testo1
    tokensPOS2 = nltk.pos_tag(Corpus2) 
    HapaxTxt1 = HapaxIncremento(Corpus1) #Invoco la funzione che incrementa gli Hapax ogni 1000 
    HapaxTxt2 = HapaxIncremento(Corpus2)
    ListaPos1 = EstraiPOS(tokensPOS) #Invoco la funzione che estrae soltanto i Pos dal testo
    ListaPos2 = EstraiPOS(tokensPOS2)
    BigrammiPOS1 = bigrams(ListaPos1) #Estraggo i bigrammi dai POS
    BigrammiPOS2 = bigrams(ListaPos2)
    ProbCond1 = Condizionata(ListaPos1,BigrammiPOS1) #Invoco la funzione che mi calcola la Probabilità Condizionata sui Pos e sui Bigrammi
    ProbCond2 = Condizionata(ListaPos2,BigrammiPOS2)
    LMI1 = CalcoloLMI(ListaPos1,BigrammiPOS1) #Invoco la funzione che calcola la Forza Associativa Massima (in termini di Local Mutual Information) sui Pos e sui Bigrammi
    LMI2 = CalcoloLMI(ListaPos2,BigrammiPOS2)

    #STAMPA I VARI CONFRONTI
    

    #IL NUMERO TOTALE DELLE FRASI
    
    print "\n\n- NUMERO TOTALE  DELLE FRASI -\n"
    NumeroFrasi1 = len(frasi1)
    NumeroFrasi2 = len(frasi2)
    print "Il numero totale delle frasi presenti nel testo 1 (",file1,") sono:", NumeroFrasi1
    print "Il numero totale delle  frasi presenti nel testo 2 (",file2,") sono:", NumeroFrasi2

    if(NumeroFrasi1>NumeroFrasi2): #confronta il valore più grande
        print file1, "ha più frasi di", file2
    elif(NumeroFrasi1<NumeroFrasi2):
        print file2, "ha più frasi di", file1
    else:
        print "i due file hanno lo stesso numero di frasi"

    #IL NUMERO TOTALE DEI TOKEN
    
    print "\n\n- NUMERO TOTALE DEI TOKEN -\n"
    LunghezzaCorpus1 = len(Corpus1)
    LunghezzaCorpus2 = len(Corpus2)
    print "Il testo 1 (", file1,") ha", LunghezzaCorpus1, "tokens totali"
    print "Il testo 2 (", file2,") ha", LunghezzaCorpus2, "tokens totali"
    if(LunghezzaCorpus1>LunghezzaCorpus2): #confronto il valore più grande
        print file1, "ha più tokens di", file2
    elif(LunghezzaCorpus1<LunghezzaCorpus2):
        print file2, "ha più tokens di", file1
    else:
        print "i due file hanno lo stesso numero di tokens"


    #LUNGEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN

    print "\n\n- LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN -\n"
    LunghezzaMediaFrasi1 = LunghezzaCorpus1*1.0/NumeroFrasi1*1.0
    LunghezzaMediaFrasi2 = LunghezzaCorpus2*1.0/NumeroFrasi2*1.0
    print "La lunghezza media delle frasi in termini di token del testo 1 (",file1,") è:", LunghezzaMediaFrasi1
    print "La lunghezza media delle frasi in termini di token del testo 2 (",file2,") é:", LunghezzaMediaFrasi2

    if(LunghezzaMediaFrasi1>LunghezzaMediaFrasi2): #confronta il valore più grande
        print file1, "ha frasi più lunghe di", file2,"\n"
    elif(LunghezzaMediaFrasi1<LunghezzaMediaFrasi2):
        print file2, "ha frasi più lunghe di", file2, "\n"
    else:
        print "i due file hanno in media le frasi lunghe uguale"

    #LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERI

    print "\n\n- LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERI -\n"
    NumeroCaratteri1 = NumeroCaratteri(Corpus1)
    NumeroCaratteri2 = NumeroCaratteri(Corpus2)
    LunghezzaMediaParole1 = NumeroCaratteri1*1.0/LunghezzaCorpus1*1.0
    LunghezzaMediaParole2 = NumeroCaratteri2*1.0/LunghezzaCorpus2*1.0
    print "la lunghezza media delle parole in termini di token del testo 1 (",file1,") è:", LunghezzaMediaParole1
    print "La lunghezza media delle parole in termini di token del testo 2 (",file2,") è:", LunghezzaMediaParole2

    if(LunghezzaMediaParole1>LunghezzaMediaParole2): #confronta il valore più grande
        print file1, "ha parole più lunghe di", file2
    elif(LunghezzaMediaParole1<LunghezzaMediaParole2):
        print file2, "ha parole più lunghe di", file1
    else:
        print "i due file hanno in media le parole lunghe uguale"
        
    #GRANDEZZA VOCABOLARIO

    print "\n\n- GRANDEZZA VOCABOLARIO - \n"
    Vocabolario1 = set(Corpus1)
    Vocabolario2 = set(Corpus2)
    print "La Grandezza del Vocabolario del File ", file1, "è di ", len(Vocabolario1), "tokens"
    print "La Grandezza del Vocabolario del File", file2, "è di ", len(Vocabolario2), "tokens"
    

 
    #INCREMENTO HAPAX
    
    print "\n\n- INCREMENTO DEGLI HAPAX OGNI 1000 TOKEN - \n"
    print file1, "\t", file2
    for elem1, elem2 in zip(HapaxTxt1, HapaxTxt2):
        print "- %-20s" % (elem1), "- %-20s" % (elem2)


    #RAPPORTO SOSTANTIVI/VERBI
    
    print "\n\n- RAPPORTO SOSTANTIVI/VERBI -\n"
    Rapporto1 = RapportoSostantiviVerbi(tokensPOS)
    Rapporto2 = RapportoSostantiviVerbi(tokensPOS2)
    print "IL RAPPORTO SOSTANTIVI/VERBI NEL TESTO 1 (",file1,") E': ", Rapporto1
    print "IL RAPPORTO SOSTANTIVI/VERBI NEL TESTO 2 (",file2,") E': ", Rapporto2

    #LE 10 POS PIU FREQUENTI
    
    POSPiuFrequenti = POSFrequenti(tokensPOS)
    POSPiuFrequenti2 = POSFrequenti(tokensPOS2)
    print "\n\n- LE 10 POS PIU' FREQUENTI DEL TESTO 1 (",file1,") -\n"
    for elem in POSPiuFrequenti:
        print "POS---> " + str(elem[0]) + "\tFrequenza---> " + str(elem[1])
    print "\n\n- LE 10 POS PIU' FREQUENTI DEL TESTO 2 (",file2,") -\n"
    for elem in POSPiuFrequenti2:
        print "POS---> " + str(elem[0]) + "\tFrequenza---> " + str(elem[1])

    #I 10 POS TAG CON PROBABILITA' CONDIZIONATA MASSIMA
    
    print "\n\n- I 10 BIGRAMMI DI POS TAG CON PROBABILITA' CONDIZIONATA MASSIMA -\n"
    print "\t\t\t", file1, "\t\t\t\t\t\t\t\t\t\t\t", file2
    for elemento1, elemento2 in zip(ProbCond1, ProbCond2):
        print "Bigramma --->%-3s - %-20s  Prob. Condizionata ---> %-0s - %-20s"  % (elemento1[1][0],elemento1[1][1], "%1.2f" % elemento1[0], "%"), "Bigramma ---> %-3s - %-20s Prob. Condizionata ---> %-0s %-20s" % (elemento2[1][0], elemento2[1][1], "%1.2f" % elemento2[0], "%")

    #I 10 BIGRAMMI DI POS CON FORZA ASSOCIATIVA MASSIMA (CALCOLA IN TERMINI DI LOCAL MUTUAL INFORMATION)
    
    print "\n\n- I 10 BIGRAMMI DI POS CON FORZA ASSOCIATIVA MASSIMA (CALCOLATA IN TERMINI DI LOCAL MUTUAL INFORMATION) DEL TESTO 1 (",file1,")  -\n"
    for Bigramma1 in LMI1:
        print "Bigramma---> " + str(Bigramma1[1]) + "\tForza Associativa---> " + str(Bigramma1[0])
    print "\n\n- I 10 BIGRAMMI DI POS CON FORZA ASSOCIATIVA MASSIMA (CALCOLATA IN TERMINI DI LOCAL MUTUAL INFORMATION) DEL TESTO 2 (",file2,") -\n"
    for Bigramma2 in LMI2:
       print "Bigramma---> " + str(Bigramma2[1]) + "\tForza Associativa---> " + str(Bigramma2[0]) 


main(sys.argv[1], sys.argv[2])
    



