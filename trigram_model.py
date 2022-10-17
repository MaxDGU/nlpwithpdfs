import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else word for word in sequence]
                else: 
                    yield sequence



def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


#part 1 -extracting n-grams from a sentence:

def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    #assuming the sequence doesn't already have START/STOP tokens, insert:
    
    start = "START"
    end = "STOP"
    gram = []
    if n == 1:
        sequence.insert(len(sequence), end)
    else:
        sequence.insert(len(sequence), end)
        for i in range(n-1):
           sequence.insert(0, start)   #needs to return list of tuples 

    for i in range(len(sequence)-(n-1)):
        gram.append(tuple(sequence[i:i+n]))        

    return gram


#how to get rid of digits here? 
class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.probtotal = 0.0

        #iterate through corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        #self.wordtotal = len(self.unigramcounts)


#part 2, n-gram frequencies in corpus
    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries/Counters of unigram, bigram,
        and trigram counts. 
        """
        ##Your code here
        unigrams = []
        bigrams = []
        trigrams = []
        start = "START"
        end = "STOP"
        self.totalwordcount = 0 #keep track of total word count (including all duplicates)
        self.sentencecount = 0 #total # of sentences
        for sentence in corpus:

            self.sentencecount += 1 
            #unigrams first
            #sentence.insert(len(sentence), end)
            for word in range(len(sentence)):
                if not sentence[word].isdigit():
                    unigrams.append((sentence[word],))
                    self.totalwordcount += 1
            #bigrams
            #sentence.insert(0, start)
            for word in range(len(sentence)-1):

                if not sentence[word].isdigit() and not sentence[word+1].isdigit():
                    bigrams.append((sentence[word], sentence[word+1]))
            #trigrams
            #sentence.insert(0, start)
            for word in range(len(sentence)-2):
                if not sentence[word].isdigit() and not sentence[word+1].isdigit() and not sentence[word+2].isdigit() :
                    trigrams.append((sentence[word], sentence[word+1], sentence[word+2]))

        self.unigramcounts = Counter(unigrams)
        self.bigramcounts = Counter(bigrams)
        self.trigramcounts = Counter(trigrams)   
       
        return self.unigramcounts, self.bigramcounts, self.trigramcounts

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """
        nom = self.trigramcounts[trigram]
        subgram = trigram[:2]
        #print(subgram)

        #when there are 2 start instances, denom = all sentences 
        if (subgram == ('START', 'START')):
            denom = self.sentencecount
            prob = (nom/denom) 

        #when there is 1 start instance, denom = count of the first word after 'START'
        elif (trigram[0] == 'START' and trigram[1] != 'START'):
            denom = self.bigramcounts[subgram]
            if (denom == 0):
                prob = 0 
                return prob
            else:
                prob = (nom/denom) 

        else:
            denom = self.bigramcounts[subgram]
            if (denom == 0):
                #unseen, estimate w/ unigram
                if (subgram[0] == 'START'):
                    denom = self.unigramcounts[(subgram[1]),]
                else:
                    denom = self.unigramcounts[(subgram[0]),]
                prob = (nom/denom)
            else:
                prob = (nom/denom) 
        

        return prob

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        count = self.bigramcounts[bigram]

        subgram = bigram[0] 
        if (subgram == 'START',):
            denom = self.sentencecount
        else:
            denom = self.unigramcounts[(subgram),]
        
        prob = (count/denom)

        return prob
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        count = self.unigramcounts[(unigram),] #assume the comma ',' is NOT given after the word 

        denom = self.totalwordcount
        prob = (count/denom)
        return prob

 
    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0


        prob = lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability((trigram[1], trigram[2])) + lambda3*self.raw_unigram_probability(trigram[2])

    
     
        return prob
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        total = 0
        trigrams = get_ngrams(sentence,3) #get trigrams
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram) #get trigram probs
            log = math.log2(prob)
            total += log

        return total #check if this works (not vetted)

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        totalwords = 0
        probtotal = 0 
        for sentence in corpus:
            logprob = self.sentence_logprob(sentence)
            probtotal += logprob 
            #calculate total word tokens in test corpus
            for i in sentence:
                totalwords += 1
            totalwords -= 2 #to exclude the 2 START tokens in trigram (keeping STOP)

        l = (probtotal)/(totalwords)
        perp = 2**(-l)

        return perp


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1) #high model
        model2 = TrigramModel(training_file2) #low model

        total = 0
        correct = 0       
        #compare perplexities, return accuracy (correct predictions/total predictions)

        #high essays
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp < pp2: #we expect pp to be more accurate for high essays
                correct +=1
            
     #low essays
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            if pp < pp2: #we expect pp to be more accurate for low essays
                correct +=1
            
        
        return (correct/total)

#essay_scoring_experiment("train_high.txt", "train_low.txt", "/Users/SGupta/Desktop/hw1/ets_toefl_data/test_high", "/Users/SGupta/Desktop/hw1/ets_toefl_data/test_low")

if __name__ == "__main__":

    #testing get_ngrams, Part 1
   # n = get_ngrams(["smallest", "man", "in", "Colombia"],2)
   # print(n)

    #test count_ngrams Part 2
    model1 = TrigramModel("swift1.txt") 
    model2 = TrigramModel("swift2.txt")
    model3 = TrigramModel("swift3.txt")

    
    keyword = 'money'
    uni_count = Counter(model1.unigramcounts)
    ex1= uni_count[(keyword),]

    uni_count2 = Counter(model2.unigramcounts)
    ex2 = uni_count2[(keyword),]
    uni_count3 = Counter(model3.unigramcounts)
    ex3 = uni_count3[(keyword),]

    candidates=[ex1,ex2,ex3]
    print(candidates)
    print(min(candidates))
    

    tri_count = Counter(model1.trigramcounts)
    top_ten = tri_count.most_common(10)
    #tri2 = Counter(model2.trigramcounts)
    #top2 = tri2.most_common(10)
    #tri3 = Counter(model3.trigramcounts)
    #top3 = tri3.most_common(10)
    #print(top_ten)
    #print(top2)
    #print(top3)

   
    #print(model.raw_trigram_probability(('transfer', 'to', 'uk')))
    #print(model.raw_unigram_probability(('uk')))
    #print(model.raw_unigram_probability(('usa')))






