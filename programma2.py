# -*- coding: utf-8 -*-

import sys
import codecs
import nltk
import re

#Funzione che divide le frasi in Token

def CalcolaToken(frasi): 
    tokensTOT=[]
    for frase in frasi:
        tokens=nltk.word_tokenize(frase)             #divido la frase in token
        tokensTOT=tokensTOT+tokens                   #creo la lista che contiene tutti i token del testo
    return tokensTOT                                 #restituisco i risultati


#Funzione che estrae i 10 nomi Propri/di persona più frequenti all'interno del testo

def NomiPropri(n):
    listanomi = []                    #Lista dove verranno inseriti i nomi
    for frase in n:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        analisi = nltk.ne_chunk(tokensPOS)              #Applico la Named Entity Chunk per estrapolare le informazioni
        IOBformat = nltk.chunk.tree2conllstr(analisi)    #Trasformo l'albero in formato IOB
        for nodo in analisi:                             #Ciclo la lista dei Chunk
            NE = ""
            if hasattr(nodo, 'label'):                   #Controlla se Chunk è un nodo intermedio e non una foglia
                if nodo.label() in ["PERSON"]:           #Estrae l'etichetta della ne del nodo
                    for partNE in nodo.leaves():         #Estrae la lista di tutte le foglie del nodo intermedio, ciclo la lista per trovare che ilnome di persona si trova all'indice 0
                        NE = NE + ' ' + partNE[0]        #Concateno la foglia alla lista Ne
                    listanomi.append(NE)                 #Appendo il risultato ad una lista finale
        frequenza = nltk.FreqDist(listanomi)           #Calcolo la frequenza dei Nomi
        nomif = frequenza.most_common(10)              #Prendo soltanto i 10 più frequenti
    return nomif                                       #Restituisco il risultato finale

#Funzione che estrae i 10 nomi di luoghi più frequenti

def Luoghi(l):
    Luoghi = []                                  #Lista dove inserisco i luoghi
    for frase in l:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        Analisi = nltk.ne_chunk(tokensPOS)
        IOBformat = nltk.chunk.tree2conllstr(Analisi)
        for nodo in Analisi:
            NE = ""
            if hasattr(nodo, 'label'):
                if nodo.label() in ["GPE"]:
                    for partNE in nodo.leaves():
                        NE = NE + ' ' + partNE[0]
                    Luoghi.append(NE)
        Frequenza = nltk.FreqDist(Luoghi)                  #Calcolo la frequenza dei luoghi più frequenti
        Luoghif = Frequenza.most_common(10)                 #Prendo soltanto i 10 più frequenti
    return Luoghif                                          #restiuisco il risultato finale
                    


#Funzione che estrae le frasi in cui compaiono i Nomi Propri e le frasi più lunghe e più corte in cui compaiono

def Estrazionefrasi(frasi):
    Nomi = NomiPropri(frasi)                              #Richiamo la funzione che estrae i nomi propri
    ListaNF = []                                          #Lista vuota che conterrà i Nomi e le frasi
    ListaX = []                                         #Lista che conterrà solo le frasi in cui compariranno i nomi propri
    List = {}
    listafinale = {}                                       #Dizionario che conterrà la lista finale che 

    for frase in frasi:                                  #Avviene il ciclo per confrontare le frasi e verificare che all'interno ci siano i nomi propri
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        analisi = nltk.ne_chunk(tokensPOS)
        for nodo in analisi:
            NE = ""
            if hasattr(nodo, 'label'):
                if nodo.label() in ["PERSON"]:
                    for partNE in nodo.leaves():
                        NE = NE+' '+partNE[0]
                    for elem in Nomi:
                        if elem[0] == NE:
                            ListaNF = ListaNF + [(elem[0], frase)]             #Lista che conterrà i nomi e le rispettive frasi
    for (nome, frase) in ListaNF:
        ListaX = ListaX + [(frase)]                                            #Lista che conterrà soltanto le frasi in cui compaiono i nomi propri
                            

    for (n, f) in ListaNF:                                                  
        List.setdefault(n, []).append(f)                                     #Inserisco i nomi e le frasi in un Dizionario 
    for x in List:                                                           #Avviene il ciclo per controllare la lunghezza delle frasi
        LunghezzaMAX = 0
        LunghezzaMin = 0
        for y in List[x]:
            Parole = nltk.word_tokenize(y)
            Lunghezza = len(Parole)
            if Lunghezza > LunghezzaMAX:
                LunghezzaMAX = Lunghezza
                FraseMAX = y
            if LunghezzaMin == 0:
                LunghezzaMin = Lunghezza
                FraseMin = y
            elif Lunghezza < LunghezzaMin:
                LunghezzaMin = Lunghezza
                FraseMin = y
        listafinale.setdefault(x,[(FraseMin, FraseMAX)])            #Appendo al dizionario finale la frase minima e massima in cui compaiono i nomi propri
    return listafinale, ListaX                                      #Restituisco come primo risultato il dizionario con la frase minima e massima e come secondo risultato la lista delle frasi in cui compare il nome proprio

#Funzione che prende i sostantivi più frequenti

def Sostantivi(testoPOS):
     ListaS = []                                            #Lista in cui inserisco i sostantivi
     for (tok, pos) in testoPOS:
         if pos in ["NN", "NNS", "NNP", "NNPS"]:          #controllo che il pos sia uguale a quello dei sostantivi
             ListaS.append(tok)                           #Appendo alla Lista i sostantivi, escludendo il pos
         Freq = nltk.FreqDist(ListaS)                     #Calcolo la frequenza dei sostantivi
         sost = Freq.most_common(10)                       #Prendo i 10 sostantivi più frequenti
     return sost                                           #Restituisco il risultato finale
  
 
#Funzione che prende i Verbi più frequenti

def Verbi(testoPOS):
    ListaV = []                                                  #Lista in cui inserisco i verbi
    for (tok, pos) in testoPOS:
        if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:       #controllo che il pos sia uguale a quello dei verbi
            ListaV.append(tok)                                      #Appendo alla lista i verbi, escludendo il pos
        Freq = nltk.FreqDist(ListaV)                                #Calcolo la frequenza dei verbi
        verb = Freq.most_common(10)                                  #Prendo i 10 verbi più frequenti
    return verb                                                       #restituisco il risultato finale

#Funzione che verifica che la lunghezza delle frasi in cui compaiono i nomi propri sia minimo di 8 token e massimo di 12

def LunFrasi(Frasi, Corpus):
    f = []
    Corpusf = []
    for frase in Frasi:
        token = nltk.word_tokenize(frase)                    #Tokenizzo le frasi
        Corpusf.append(token)                                #Appendo alla lista la frase intera e non i singoli token
    for frase in Corpusf:                                    #Ciclo ogni frase tokenizzata
        Lungh = len(frase)
        if(Lungh>=8) and (Lungh<=12):                        #Verifico che le frasi siano composte da minimo 8 token e da massimo 12
            f.append(frase)                                   #Appendo alla lista le frasi che soddisfanno la condizione
    return f                                                  #restiuisco il risultato finale

#Funzione che calcola il Markov di Ordine 0

def Markov0(Lunf, LungCorpus, Testo):
    ProbMAX = 0                                 #Valore indicativo 
    Freq = nltk.FreqDist(Testo)                #Conto la frequenza dei Token all'interno del Corpus
    for frase in Lunf:                         #Scansiono tutte le frasi che soddisfanno le condizioni della precedente funzione
        result = []
        Parole = []
        ProbFrase = 1.0                                                   #Imposto la Probabilità con la virgola, in modo che poi mi dia sempre un risultato con la virgola
        for tok in frase:                                                 #Per ogni frase viene scansionato ogni token
            ProbToken = (Freq[tok] * 1.0 / LungCorpus* 1.0)               #Calcolo la probabilità dei Token
            ProbFrase = ProbFrase * ProbToken                              #Calcolo la probabilità della frase moltiplicando la probabilità dei Token per la probabilità della frase data dai token precedenti
            Parole = tok, Freq[tok]                                         #Insieme dei Valori
            result.append(Parole)                                           #Appendo alla lista i risultati
        if ProbFrase > ProbMAX:                                             #Verifica qual'è la frase con probabilità massima
            ProbMAX = ProbFrase                                             #Probabilità massima prende il valore della probabilità della frase in considerazione
            FraseMAX = frase                                                #Frase più probabile
    return "\nFRASE:\n",FraseMAX, "\nPROBABILITA':\n",ProbMAX                #restituisco i risultati
  

#FUNZIONE PRINCIPALE

def main(file1, file2):
    FileInput = codecs.open(file1, "r", "utf-8")                               #prende il file codificato in utf-8
    FileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = FileInput.read()                                                   #viene letto il file e viene assegnato tutto il suo contenuto una variabile di tipo string
    raw2 = FileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')        #viene caricato il modello statistico 'english.pickle' su cui ci si basa per dividere il testo in frasi
    Frasi1 = sent_tokenizer.tokenize(raw1)                                     #Divido le frasi in token
    Frasi2 = sent_tokenizer.tokenize(raw2)
    en1, f1 = Estrazionefrasi(Frasi1)                                          #La prima variabile restiuisce le frasi più lunghe e più corte in cui compare il nome proprio, la seconda variabile soltanto la lista delle frasi in cui compare ilnome proprio
    en2, f2 = Estrazionefrasi(Frasi2)
    EL1 = Luoghi(f1)                                                            #Invoco la funzione che estrae i luoghi e la applico alla lista delle frasi in cui compare il nome proprio
    EL2 = Luoghi(f2)
    Persone1 = NomiPropri(f1)                                                     #Invoco la funzione che estrare i nomi di persona e la applico alla lista delle frasi in cui compare il nome proprio
    Persone2 = NomiPropri(f2)
    TestoToken1 = CalcolaToken(f1)                                                 #Calcolo i Token sulle frasi in cui compare il nome proprio
    TestoToken2 = CalcolaToken(f2)
    Corpus1 = CalcolaToken(Frasi1)                                                  #Calcolo i Token su tutto il corpus
    Corpus2 = CalcolaToken(Frasi2)
    POS1 = nltk.pos_tag(TestoToken1)                                                #Applica ad ogni token preso dalle frasi in cui compare il nome Proprio il tag di una parte del discorso
    POS2 = nltk.pos_tag(TestoToken2)
    Sostantivi1 = Sostantivi(POS1)                                                   #Invoco la funzione che estrae i sostantivi e la applico ai POS per le frasi in cui compare il nome proprio
    Sostantivi2 = Sostantivi(POS2)
    Verbi1 = Verbi(POS1)                                                             #Invoco la funzione che estrae i verbi e la applico ai POS per le frasi in cui compare il nome proprio
    Verbi2 = Verbi(POS2)
    LunghezzaFrasi1 = LunFrasi(f1, Corpus1)                                           #Calcola che la lunghezza delle frasi in cui compare il nome proprio sia composta da minimo 8 token e massimo 12
    LunghezzaFrasi2 = LunFrasi(f2, Corpus2)
    LunghezzaTesto1 = len(Corpus1)                                                     #Calcola la lunghezza di tutto il corpus
    LunghezzaTesto2 = len(Corpus2)
    Markov01 = Markov0(LunghezzaFrasi1,LunghezzaTesto1, Corpus1)                       #Invoco la funzione che calcola il Markov di Ordine 0 sulle frasi in cui compare il nome proprio
    Markov02 = Markov0(LunghezzaFrasi2,LunghezzaTesto2,Corpus2)
  
    
    #STAMPE RISULTATI
 
    #FRASE PIU' LUNGA E PIU' CORTA CHE CONTENGONO I 10 NOMI PROPRI PIU' FREQUENTI
    
    print "\n\n- FRASE PIU' CORTA E PIU' LUNGA CHE CONTENGONO I 10 NOMI PROPRI PIU' FREQUENTI PER IL TESTO 1 (",file1,") -\n"
    for x in en1:
        print "NOME--->",(x).encode('utf-8') + "\n"
        for z, o in en1[x]:
            print "FRASE CORTA:",(z).encode('utf-8') + "\n"
            print "FRASE LUNGA:",(o).encode('utf-8') + "\n"

    print "\n\n- FRASE PIU' CORTA E PIU' LUNGA CHE CONTENGONO I 10 NOMI PROPRI PIU' FREQUENTI PER IL TESTO 2 (",file2,") -\n"
    for x in en2:
        print "NOME--->",(x).encode('utf-8') + "\n"
        for z, o in en2[x]:
            print "FRASE CORTA:",(z).encode('utf-8') + "\n"
            print "FRASE LUNGA:",(o).encode('utf-8') + "\n"

    #LUOGHI 

    print "\n\n- I LUOGHI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 1 (",file1,") -\n"
    for luogo1 in EL1:
        print "\nLUOGO:\t" + (luogo1[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(luogo1[1]) + "\n"

    print "\n\n- I LUOGHI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 2 (",file2,") -\n"
    for luogo2 in EL2:
        print "\nLUOGO:\t" + (luogo2[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(luogo2[1]) + "\n"

        
    #PERSONE
    
    print "\n\n- LE PERSONE PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 1 (",file1,") -\n"
    for p1 in Persone1:
        print "\nPERSONA:\t" + (p1[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(p1[1]) + "\n"

    print "\n\n- LE PERSONE PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 2 (",file2,") -\n"
    for p2 in Persone2:
        print "\nPERSONA:\t" + (p2[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(p2[1]) + "\n"

        
    #SOSTANTIVI
    
    print "\n\n- I SOSTANTIVI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 1 (",file1,") -\n"
    for s1 in Sostantivi1:
        print "\nSOSTANTIVO:\t" + (s1[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(s1[1]) + "\n"
        
    print "\n\n- I SOSTANTIVI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 2 (",file2,") -\n"
    for s2 in Sostantivi2:
         print "\nSOSTANTIVO:\t" + (s2[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(s2[1]) + "\n"

         
    #VERBI

    print "\n\n- I VERBI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 1 (",file1,") -\n"
    for v1 in Verbi1:
        print "\nVERBO:\t" + (v1[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(v1[1]) + "\n"

    print "\n\n- I VERBI PIU' FREQUENTI CHE SONO PRESENTI ALL'INTERNO DELLE FRASI IN CUI COMPAIONO I NOMI PROPRI NEL TESTO 2 (",file2,") -\n"
    for v2 in Verbi2:
        print "\nVERBO:\t" + (v2[0]).encode('utf-8') + "\n\nFREQUENZA:\t" + str(v2[1]) + "\n"
        

    #DATE, GIORNI, MESI ESTRATTI DALLE FRASI IN CUI COMPARE IL NOME PROPRIO

    #FILE1
    
    
    #STAMPA DATE
    
    print "\n\n- LE, DATE, I MESI E I GIORNI DELLA SETTIMANA ESTRATTI DAL TESTO 1 (",file1,") UTILIZZANDO LE ESPRESSIONI REGOLARI -\n\n"
    print "\n\n- DATE ESTRATTE -\n\n"
    ListaD = []
    for i in f1:
        ListaD = ListaD + re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12][0-9]|3[01])([\,]|[\, ])\ ([12][0-9]\d\d)|(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d',i)
    Freq = nltk.FreqDist(ListaD)
    Result = Freq.most_common()
    if not(Result==[]):
        for (a,b) in Result:
            print "DATA:\t" + str(a[0]) + "\t" + str(a[1]) + str(a[2]) + "\t" + str(a[3]) + "\tFreq:\t" + str(b) + "\n\n\n"
    elif (Result==[]):
        print "Non è stata trovata alcuna data all'interno del testo!"


    #STAMPA GIORNI
    
    print "\n\n- GIORNI ESTRATTI -\n\n"
    ListaG = []
    for g in f1:
        ListaG = ListaG + re.findall(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',g)
    Freq = nltk.FreqDist(ListaG)
    Result = Freq.most_common()
    if not(Result==[]):
        for match in Result:
            print "GIORNO:\t"+ str(match[0]) + "\t\t" + "FREQUENZA:\t" + str(match[1]) + "\n\n\n"
    elif (Result==[]):
        print "Non è stato trovato alcun giorno all'interno del testo!"
  
    #STAMPA MESI
    
    print "\n\n- MESI ESTRATTI -\n\n"
    ListaM = []
    for m in f1:
        ListaM = ListaM + re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)',m)
    Freq = nltk.FreqDist(ListaM)
    Result = Freq.most_common()
    if not(Result==[]):
        for match in Result:
            print "MESE:\t" + str(match[0]) + "\t\t" + "FREQUENZA:\t" + str(match[1]) + "\n\n\n"
    elif (Result==[]):
        print "Non è stato trovato alcun mese all'interno del testo!"


    #FILE 2
    
  
    #STAMPA DATE
    
    print "\n\n- LE DATE, I MESI E I GIORNI DELLA SETTIMANA ESTRATTI DAL TESTO 2 (",file2,") UTILIZZANDO LE ESPRESSIONI REGOLARI -\n\n"
    print "\n\n- DATE ESTRATTE -\n\n"
    ListaD = []        #Lista che conterrà i risultati dell'espressione regolare relativa alla data con mese esteso
    ListaD2 = []       #Lista che conterrà i risultati dell'espressione regolare relativa alle date in formato europeo e americano
    for i in f2:       #Ciclo all'interno delle frasi che contengono il nome proprio
        ListaD = ListaD + re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12][0-9]|3[01])([\,]|[\, ])\ ([12][0-9]\d\d)',i)   #Espressione Regolare utilizzata per estrarre le date
    Freq = nltk.FreqDist(ListaD) #Calcolo la frequenza dei risultati della lista
    Result = Freq.most_common() #Lista che contiene sia i risultati dell'espressione regolare che la frequenza
    if not(Result==[]):
        for (a,b) in Result:
            print "DATA:\t" + str(a[0]) + "\t" +  str(a[1]) + str(a[2]) + "\t" + str(a[3]) + "\tFreq:\t" +  str(b) + "\n\n\n"
    elif (Result==[]):
        print "Non è stata trovata alcuna data all'interno del testo!"
  
                
    #STAMPA GIORNI
    
    print "\n\n- GIORNI ESTRATTI -\n\n"
    ListaG = [] #Lista che conterrà i risultati dell'espressione regolare
    for g in f2: #Ciclo all'interno delle frasi che contengono il nome proprio
        ListaG = ListaG + re.findall(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',g) #Espressione regolare utilizzata per estrarre i giorni
    Freq = nltk.FreqDist(ListaG) #Calcolo la frequenza dei risultati della lista
    Result = Freq.most_common() #Lista che contiene sia i risultati dell'espressione regolare che la frequenza
    if not(Result==[]):
        for match in Result:
            print "GIORNO:\t"+ str(match[0]) + "\t\t" + "FREQUENZA:\t" + str(match[1]) + "\n\n\n"
    elif (Result==[]):
        print "Non è stato trovato alcun giorno all'interno del testo!"
       
   
    
                
    #STAMPA MESI
    
    print "\n\n- MESI ESTRATTI -\n\n"
    ListaM = [] #Lista che conterrà i risultati dell'espressione regolare
    for m in f2: #Ciclo all'interno delle frasi che contengono il nome proprio
        ListaM = ListaM + re.findall(r'(January|February|March|April|May|June|July|August|September|October|November|December)',m) #Espressione regolare utilizzata per estrarre i mesi
    Freq = nltk.FreqDist(ListaM) #Calcolo la frequenza dei risultati della lista
    Result = Freq.most_common() #Lista che contiene sia i risultati dell'espressione regolare che la frequenza
    if not(Result==[]):
        for match in Result:
            print "MESE:\t" + str(match[0]) + "\t\t" + "FREQUENZA:\t" + str(match[1]) + "\n\n\n"
    elif (Result==[]):
        print "Non è stato trovato alcun mese all'interno del testo!"
 
    
    #MARKOV0
    
    print "\n\n- LA FRASE CON LA PROBABILITA' PIU' ALTA CALCOLATA ATTRAVERSO UN MARKOV DI ORDINE 0 PER IL TESTO 1 (",file1,") -\n"
    for M1 in Markov01:
        print M1,"\n"

    print "\n\n- LA FRASE CON LA PROBABILITA' PIU' ALTA CALCOLATA ATTRAVERSO UN MARKOV DI ORDINE 0 PER IL TESTO 2 (",file2,") -\n"
    for M2 in Markov02:
        print M2,"\n"

    
  
   
main(sys.argv[1],sys.argv[2])

    
