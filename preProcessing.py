from nltk.tokenize 			  import word_tokenize, sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import string
import re

class PreProcessing(object):
	def __init__(self):
		self._punctuations 	= string.punctuation 					#kumpulan punctuation
		self._nlp 			= spacy.load('en_core_web_sm') 			#load model untuk stemming & tokenizing
		self._stopwords 	= spacy.lang.en.stop_words.STOP_WORDS 	#kumpulan stopwords

	def _removing_numbers_in_sentence(self, sentence):
		return re.sub('[0-9]+', '', sentence)

	def _removing_punctuation_in_sentence(self, sentence):
		return sentence.translate(str.maketrans('','',self._punctuations))
		
	def _removing_punctuation_in_sentences_list(self, sentences):
		result = []
		for sentence in sentences:
			result.append(self._removing_punctuation_in_sentence(sentence))
		return result

	def _removing_whitespaces_in_sentence(self, sentence):
		return ' '.join(sentence.split())

	def _removing_whitespaces_in_sentences_list(self, sentences):
		result = []
		for sentence in sentences:
			result.append(self._removing_whitespaces_in_sentence(sentence))
		return result

	def _removing_whitespaces_in_word(self, word):
		return word.replace(' ','')

	def _removing_stopwords_in_words_list(self, words):
		result = []
		for word in words:
			if (word not in self._stopwords) and (word not in self._punctuations) and (self._removing_whitespaces_in_word(word) != ''):
				result.append(word)
		return result

	def _lowering_sentence(self, sentence):
		return sentence.lower()

	def _words_tokenizing(self, sentence):
		return word_tokenize(sentence)

	def _sentence_tokenizing(self, sentence):
		return sent_tokenize(sentence)

	def _lemmatization(self, sentence):
		words  = self._nlp(sentence)
		result = []
		for word in words:
			if word.is_alpha and not word.is_stop and word.lemma_ != '-PRON-':	
				if word.pos_ == 'ADJ' or word.pos_ == 'ADV' or word.pos_ == 'NOUN' or word.pos_ == 'PROPN' or word.pos_ == 'VERB':
					result.append(word.lemma_)
		return result