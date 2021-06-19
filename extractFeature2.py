#!/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import csv
from keras.preprocessing.text import text_to_word_sequence
import sys
csv.field_size_limit(sys.maxsize)
from sklearn.preprocessing import MultiLabelBinarizer
# vic_file_text = 'victor_text.csv'
# vic_file_pos = 'victor_pos.csv'

train_file = '/home/someone/Desktop/paper/error analysis/test errors/common_errors3.csv'
train_pos_file='/home/someone/Desktop/paper/error analysis/test errors/common_errors3_pos.csv'



lex = open('/home/someone/Desktop/paper/error analysis/lex.csv',encoding='utf-8')
lexicon=[]
for line in lex:
    lexicon.append(line.rstrip("\n"))

#print(lexicon)

class Tsv(object):
    delimiter = ','
    #quotechar = '"'
    escapechar = '\\'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_ALL


def splitNews(file):
    reviews = []
    labels = []
    texts = []
    lex_list=[]
    lex_c=[]
    lex_r=[]
    ids = []
    with open(file, 'r', encoding="utf-8") as r:
        reader = csv.reader(r, dialect=Tsv)
        for i, row in enumerate(reader):
            text = ''
            doc = []
            temp=[]
            for sent in row[1].split(','):
                wordTokens = text_to_word_sequence(sent, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
                doc.append(wordTokens)
                #A= set(lexicon).intersection(set(wordTokens))
                #temp= temp.union(A)
                text += sent + ' '
                for s in lexicon:
                    if (s+' ' in sent) or (s+'.' in sent) or (s+'?' in sent) or (s+'!' in sent):
                        temp.append(s)
            temp=set(temp)
            lex_list.append(list(temp))
            if len(list(temp))==0:
                lex_c.append(0)
                lex_r.append(0.0)
            else:
                ratio= round(len(list(temp))/len(sent) , 7)
                lex_c.append(1)
                lex_r.append(ratio)
            reviews.append(doc)
            texts.append(text)
            labels.append(row[2])
            ids.append(row[0])
    return texts, reviews, labels,lex_list,lex_c,lex_r,ids


# vic_texts, vic_reviewsText, viclabels,lex_list1,lex_c1,lex_r1 = splitNews(vic_file_text)
# vic_POS, vic_reviewsPOS, viclabels,lex_list2,_, _ = splitNews(vic_file_pos)

train_texts, train_reviewsText , train_labels, lex_list1,lex_c1,lex_r1, train_ids = splitNews(train_file)
train_POS, train_reviewsPOS ,_ , l1,l2,l3, _ = splitNews(train_pos_file)


def writeLex(lex_list,f):
    lx = pd.Series(lex_list)
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(lx),
                       columns=mlb.classes_,
                       index=lx.index)
    res.to_csv (f, index = None, header=True,encoding='utf-8')


#writeLex(lex_list,'lex_featuresN.csv')



# tokenizer1 = Tokenizer()
# tokenizer1.fit_on_texts(victexts)
#
# tokenizer2 = Tokenizer()
# tokenizer2.fit_on_texts(victextsPOS)

pron1=['ben','biz']
pron2=['sen','siz']
pron3=['onlar','o']
possesiveProns=['benim', 'senin', 'onun', 'bizim', 'sizin', 'onların']
verb1 = '1|unk|verb|verb' #1|unk|verb|verb|root
verb2 = '2|unk|verb|verb' #2|unk|verb|verb|root
verb3 = '3|unk|verb|verb' #3|unk|verb|verb|root
imperativeTag = 'imp|verb|verb|root'
pronOBL= 'pron|pers|obl'
pronOBJ= 'pron|pers|obj'

def featureData(text,vic_reviewsText, vic_reviewsPOS, file, labels, ids):
    othSent=[]
    impVerbs=[]
    impRatioList = []
    possPronRatioList = []
    pronVerbRatioList = []
    pronExistList=[]
    possExistList=[]
    impExistList=[]
    othRatioList =[]
    othExistList=[]
    with open(file, 'w' , encoding="utf-8") as f:
        #writer = csv.writer(w1, dialect=Tsv)
        for i, sentences in enumerate(vic_reviewsText):
            count1 = 0
            count2 = 0
            countx=0
            county=0
            counta=0
            countb=0
            temp_s1 = ''
            temp_s2 = ''
            temp_s3=''
            temp_s4 = ''
            onlyPossOrImp = ''
            possOrImpWithOthers = ''
            temp_x=''
            temp_y=''
            ox=''
            possPronSntcCnt = 0 # number of sentences that has at least one possessive pronoun
            imperativeSntcCnt = 0 # number of sentences that has an imperative verb
            pronVerbSntCnt = 0
            for j, sent in enumerate(sentences):
                sentFlag=False
                anyPossPron = False # if this sentence has poss pron
                anyImperative = False # if this sentence has imperative
                #words = (text_to_word_sequence(sent, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n'))
                #if len(list(filter(lambda x: '1|unk|PRON|Pers|nsubj' in x, tags)))>0 and filter(lambda x: verb1 in x, tags):

                #and len(list(filter(lambda x: verb1 in x, vic_reviewsPOS[i][j])))>0

                # Check if there is any possessive pronoun in the sentence
                if any(word in sent for word in possesiveProns):
                    possPronSntcCnt += 1
                    anyPossPron = True

                # Check if there is imperative in the sentence, mp|verb|verb|root -> imperative tag
                if (len(list(filter(lambda x: imperativeTag in x, vic_reviewsPOS[i][j])))>0):
                    imperativeSntcCnt += 1
                    anyImperative = True

                # if any possessive pron or imperative is used keep the sentence
                if (anyPossPron or anyImperative):
                    s = sent.copy()
                    listToStr = ' '.join(map(str, s))
                    onlyPossOrImp += listToStr + ' . '



                if ('ben' in sent or 'biz' in sent or 'bizler' in sent) and len(list(filter(lambda x: verb1 in x, vic_reviewsPOS[i][j])))>0:
                    s = sent.copy()
                    if 'ben' in sent:
                        s.insert(s.index('ben') + 1 ,'nsubj')
                    if 'biz' in sent:
                        s.insert(s.index('biz') + 1 ,'nsubj')

                    if len(list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j])))>0:
                        obls = list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j]))
                        for obl in obls:
                            a = sent[vic_reviewsPOS[i][j].index(obl)]
                            s.insert(s.index(a)+1, 'obl')
                    if len(list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j])))>0:
                        objs = list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j]))
                        for obj in objs:
                            a = sent[vic_reviewsPOS[i][j].index(obj)]
                            s.insert(s.index(a)+1, 'obj')
                    if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                        poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                        for pos in poss:
                            a = sent[vic_reviewsPOS[i][j].index(pos)]
                            s.insert(s.index(a)+1, 'nmodposs')

                    #sent.insert(k + 1, 'nsubj')
                    listToStr = ' '.join(map(str, s))
                    temp_s1 += listToStr + ' . '
                    count1 += 1
                    sentFlag=True

                elif ('sen' in sent or 'siz' in sent or 'sizler' in sent) and len(list(filter(lambda x: verb2 in x, vic_reviewsPOS[i][j])))>0:
                    s = sent.copy()
                    if 'sen' in sent:
                        s.insert(s.index('sen') + 1 ,'nsubj')
                    if 'siz' in sent:
                        s.insert(s.index('siz') + 1 ,'nsubj')

                    if len(list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j])))>0:
                        obls = list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j]))
                        for obl in obls:
                            a = sent[vic_reviewsPOS[i][j].index(obl)]
                            s.insert(s.index(a)+1, 'obl')
                    if len(list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j])))>0:
                        objs = list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j]))
                        for obj in objs:
                            a = sent[vic_reviewsPOS[i][j].index(obj)]
                            s.insert(s.index(a)+1, 'obj')
                    if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                        poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                        for pos in poss:
                            a = sent[vic_reviewsPOS[i][j].index(pos)]
                            s.insert(s.index(a)+1, 'nmodposs')

                    #sent.insert(k + 1, 'nsubj')
                    listToStr = ' '.join(map(str, s))
                    temp_s2 += listToStr + ' . '
                    count2 += 1

                elif ('o' in sent or 'onlar' in sent) and len(list(filter(lambda x: verb3 in x, vic_reviewsPOS[i][j])))>0:
                    s = sent.copy()
                    if 'o' in sent:
                        s.insert(s.index('o') + 1 ,'nsubj')
                    if 'onlar' in sent:
                        s.insert(s.index('onlar') + 1 ,'nsubj')

                    if len(list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j])))>0:
                        obls = list(filter(lambda x: '|pron|pers|obl|' in x, vic_reviewsPOS[i][j]))
                        for obl in obls:
                            a = sent[vic_reviewsPOS[i][j].index(obl)]
                            s.insert(s.index(a)+1, 'obl')
                    if len(list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j])))>0:
                        objs = list(filter(lambda x: '|pron|pers|obj|' in x, vic_reviewsPOS[i][j]))
                        for obj in objs:
                            a = sent[vic_reviewsPOS[i][j].index(obj)]
                            s.insert(s.index(a)+1, 'obj')
                    if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                        poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                        for pos in poss:
                            a = sent[vic_reviewsPOS[i][j].index(pos)]
                            s.insert(s.index(a)+1, 'nmodposs')

                    #sent.insert(k + 1, 'nsubj')
                    listToStr = ' '.join(map(str, s))
                    temp_s2 += listToStr + ' . '
                    count2 += 1

                elif ('ben' not in sent and 'biz' not in sent and 'bizler' not in sent) and len(list(filter(lambda x: verb1 in x, vic_reviewsPOS[i][j])))>0:
                    s=sent.copy()
                    xx=['sizi', 'size','sizde','sizden','sana','seni','sende','senden','sizinle','sizlerle','seninle','onunla','ona','onda','ondan','onlarla']
                    tt=list(set(xx).intersection(set(sent)))
                    if len(tt)>0:
                        for t in tt:
                            a = sent.index(t)
                            if vic_reviewsPOS[i][j][a].find('pers|obl') !=-1:
                                s.insert(s.index(t) + 1, 'obl')
                                counta +=1
                            elif vic_reviewsPOS[i][j][a].find('pers|obj') !=-1:
                                s.insert(s.index(t) + 1, 'obj')
                                counta += 1
                        if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                            poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                            for pos in poss:
                                a = sent[vic_reviewsPOS[i][j].index(pos)]
                                s.insert(s.index(a) + 1, 'nmodposs')
                        listToStr = ' '.join(map(str, s))
                        temp_s3 += listToStr + ' . '
                    else: # not to miss the sentence in case other conditions are applied for other sentences
                        listToStr = ' '.join(map(str, s))
                        #possOrImpWithOthers += listToStr + ' . '

                    # else:
                    #     listToStr = ' '.join(map(str, s))
                    #     temp_x += listToStr + ' . '
                    #     countx += 1


                elif ('sen' not in sent and 'siz' not in sent and 'sizler' not in sent) and len(list(filter(lambda x: verb2 in x, vic_reviewsPOS[i][j])))>0:
                    s=sent.copy()

                    xx=['bizi', 'bize','bizde','bizden','bana','beni','bende','benden','bizimle','benimle','bizlerle']
                    tt=list(set(xx).intersection(set(sent)))
                    if len(tt)>0:
                        for t in tt:
                            a = sent.index(t)
                            if vic_reviewsPOS[i][j][a].find('pers|obl')!=-1:
                                s.insert(s.index(t) + 1, 'obl')
                                countb+=1
                            elif vic_reviewsPOS[i][j][a].find('pers|obj')!=-1:
                                s.insert(s.index(t) + 1, 'obj')
                                countb+=1
                        if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                            poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                            for pos in poss:
                                a = sent[vic_reviewsPOS[i][j].index(pos)]
                                s.insert(s.index(a) + 1, 'nmodposs')
                        listToStr = ' '.join(map(str, s))
                        temp_s4 += listToStr + ' . '
                    else: # not to miss the sentence in case other conditions are applied for other sentences
                        listToStr = ' '.join(map(str, s))
                        #possOrImpWithOthers += listToStr + ' . '

                    # else:
                    #     listToStr = ' '.join(map(str, s))
                    #     temp_y += listToStr + ' . '
                    #     county += 1

                elif ('o' not in sent and 'onlar' not in sent) and len(list(filter(lambda x: verb3 in x, vic_reviewsPOS[i][j])))>0:
                    s=sent.copy()
                    xx = ['bizi', 'bize', 'bizde', 'bizden', 'bana', 'beni', 'bende', 'benden', 'bizimle', 'benimle','bizlerle']
                    tt = list(set(xx).intersection(set(sent)))
                    if len(tt) > 0:
                        for t in tt:
                            a = sent.index(t)
                            if vic_reviewsPOS[i][j][a].find('pers|obl')!=-1:
                                s.insert(s.index(t) + 1, 'obl')
                                countb +=1
                            elif vic_reviewsPOS[i][j][a].find('pers|obj')!=-1:
                                s.insert(s.index(t) + 1, 'obj')
                                countb += 1
                        if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))) > 0:
                            poss = list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i][j]))
                            for pos in poss:
                                a = sent[vic_reviewsPOS[i][j].index(pos)]
                                s.insert(s.index(a) + 1, 'nmodposs')
                        listToStr = ' '.join(map(str, s))
                        temp_s4 += listToStr + ' . '
                    else: # not to miss the sentence in case other conditions are applied for other sentences
                        listToStr = ' '.join(map(str, s))
                        #possOrImpWithOthers += listToStr + ' . '

                    # else:
                    #     listToStr = ' '.join(map(str, s))
                    #     temp_y += listToStr + ' . '
                    #     county += 1
                elif (anyPossPron or anyImperative): # not to miss the sentence in case other conditions are applied for other sentences
                    s = sent.copy()
                    listToStr = ' '.join(map(str, s))
                    possOrImpWithOthers += listToStr + ' . '

                # if count1>0 or count2>0:
                #     ox = temp_s1 + temp_s2 + temp_s3 +temp_s4 +temp_x +temp_y
                #     othSent.append(ox)
            count_oth=0
            if count1 > 0 and count2 > 0:  # ben soyledim. sen dinlemedin. (seni dinlemedim. beni dinlemedin.)  !!! Count1>0 AND Count2>0
                ox = temp_s1 + temp_s2 + temp_s3 + temp_s4 + onlyPossOrImp
                count_oth+=1
                pronVerbSntCnt = count1 + count2 + counta + countb
            elif count1 == 0 and count2 > 0 and counta > 0:  # sen soyledin. seni dinlemedim. !!! count2 > 0 and counta > 0:
                ox = temp_s2 + temp_s3 + temp_s4 + onlyPossOrImp
                pronVerbSntCnt = count1 + count2 + counta + countb
            elif count1 > 0 and count2 == 0 and countb > 0:  # ben soyledim. beni dinlemedin.  !!! Count1 > 0 AND Countb>0
                ox = temp_s1 + temp_s3 + temp_s4 + onlyPossOrImp
                pronVerbSntCnt = count1 + count2 + counta + countb
            elif counta > 0 and countb > 0:  # bana soyledin. seni dinlemedim. !!!  counta > 0 and countb > 0
                ox = temp_s3 + temp_s4 + onlyPossOrImp
                pronVerbSntCnt = count1 + count2 + counta + countb
            else:
                ox = onlyPossOrImp

            othSent.append(ox)

            #f.write('%s\n' % ox)

            #if ox!='':
                #sent_text = nltk.sent_tokenize(ox)
                #writer.writerow(["".join(str(ids[i])), ",".join(sent_text), "".join(labels[i])])
                #f.write('%s\n' % ox)
            #else:
                #f.write('%s\n' % (ox + text[i]))


            impRatio = round(imperativeSntcCnt / len(sentences), 7)  # ratio of sentences which has imperative verb
            impRatioList.append(impRatio)
            if impRatio != 0.0:
                impExistList.append(1)
            else:
                impExistList.append(0)

            possProsRatio = round(possPronSntcCnt / len(sentences), 7)  # ratio of sentences which has possessive pronoun
            possPronRatioList.append(possProsRatio)

            if possProsRatio != 0.0:
                possExistList.append(1)
            else:
                possExistList.append(0)

            pronVerbSntRatio = round(pronVerbSntCnt / len(sentences), 7)
            pronVerbRatioList.append(pronVerbSntRatio)

            if pronVerbSntRatio != 0.0:
                pronExistList.append(1)
            else:
                pronExistList.append(0)

            othSntRatio = round(count_oth / len(sentences), 7)
            othRatioList.append(othSntRatio)

            if othSntRatio != 0.0:
                othExistList.append(1)
            else:
                othExistList.append(0)


        # bizim -> nmodpross
        # 3|unk|NOUN|Noun|nmodposs|6|7 -> onlarin
        # ayni cumleleri yazmasin diye yeniden dolasmak??
        # if len(list(filter(lambda x: '|pron|pers|nmodposs|' in x, vic_reviewsPOS[i])))>0:
        #     for ts, tags in enumerate(vic_reviewsPOS[i]):
        #         if 'benim' in vic_reviewsText[i][ts] and vic_reviewsPOS[i][ts][vic_reviewsText[i][ts].index('benim')].find('1|unk|pron|pers|nmodposs|')!=-1 :
        #f.write("\n".join(othSentAll))

    return othSent, impRatioList, possPronRatioList, pronVerbRatioList, pronExistList, possExistList, impExistList, othRatioList,othExistList

othSent1 , impRatioList1, possPronRatioList1, pronVerbRatioList1, pronExistList1, possExistList1, impExistList1, othRatioList1,othExistList1= featureData(train_texts,train_reviewsText, train_reviewsPOS, '/home/someone/Desktop/paper/error analysis/test error counts/asd.txt', train_labels, train_ids)


#lex_c = lex_c1 + lex_c2
#lex_r = lex_r1 + lex_r2


#df_lex_text= pd.DataFrame({'lexExist':lex_c, 'lexRatio':lex_r})
#df_lex_text.to_csv(r'lex_infoN.csv',index=None, header=True,encoding='utf-8')

df_docs_feature_train = pd.DataFrame({'ID': train_ids,'Class': train_labels, 'ImperativeRatio': impRatioList1, 'PossPronRatio':possPronRatioList1, 'PronVerbRatio':pronVerbRatioList1, 'LexRatio': lex_r1,  'ImperativeExist': impExistList1, 'PossExist':possExistList1, 'PronExist':pronExistList1,'OthExist':othExistList1, 'LexExist':lex_c1})
df_docs_feature_train.to_csv (r'/home/someone/Desktop/paper/error analysis/test error counts/common_error_counts.csv', index = None, header=True,encoding='utf-8')






#df_docs_feature_train = pd.DataFrame({'News': othSent1, 'ImperativeRatio': impRatioList1, 'PossPronRatio':possPronRatioList1, 'PronVerbRatio':pronVerbRatioList1,'OthRatio':othRatioList1, 'LexRatio':lex_r1,   'ImperativeExist': impExistList1, 'PossExist':possExistList1, 'PronExist':pronExistList1,'OthExist':othExistList1,'LexExist':lex_c1, 'Class': trainlabels})

#df_docs_feature_train.to_csv (r'train_features1_text.csv', index = None, header=True,encoding='utf-8')

#df_docs_feature_test = pd.DataFrame({'News': othSent2, 'ImperativeRatio': impRatioList2, 'PossPronRatio':possPronRatioList2, 'PronVerbRatio':pronVerbRatioList2,'OthRatio':othRatioList2, 'LexRatio':lex_r2,   'ImperativeExist': impExistList2, 'PossExist':possExistList2, 'PronExist':pronExistList2,'OthExist':othExistList2, 'LexExist':lex_c2, 'Class': testlabels})

#df_docs_feature_test.to_csv (r'test_features1_text.csv', index = None, header=True,encoding='utf-8')



# othSent, impRatioList, possPronRatioList = featureData(vic_reviewsText, vic_reviewsPOS, 'vic_feature.txt')

#othSent1 , impRatioList1, possPronRatioList1= featureData(train_reviewsText, train_reviewsPOS, 'train_feature2.txt')
#othSent2 , impRatioList2, possPronRatioList2= featureData(test_reviewsText, test_reviewsPOS, 'test_feature2.txt')


#df_docs_feature_train = pd.DataFrame({'News': othSent1, 'ImperativeRatio': impRatioList1, 'PossPronRatio':possPronRatioList1, 'LexExist':lex_c1,'LexRatio':lex_r1,'Class': trainlabels})

#df_docs_feature_train.to_csv (r'train_features2_text.csv', index = None, header=True,encoding='utf-8')

#df_docs_feature_test = pd.DataFrame({'News': othSent2, 'ImperativeRatio': impRatioList2, 'PossPronRatio':possPronRatioList2, 'LexExist':lex_c2,'LexRatio':lex_r2,'Class': testlabels})

#df_docs_feature_test.to_csv (r'test_features2_text.csv', index = None, header=True,encoding='utf-8')

# othSent1 = featureData(train_reviewsText, train_reviewsPOS, 'train_feature.txt')
# othSent2 = featureData(test_reviewsText, test_reviewsPOS, 'test_feature.txt')

#print(othSent)


