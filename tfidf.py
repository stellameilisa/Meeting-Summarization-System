import math
import numpy
from preProcessing import PreProcessing

class TFIDF(object):
    def __init__(self, length = 50, tf = 1, idf = 1):
        print("tf: ", tf, " idf: ", idf)
        self._pre    = PreProcessing()
        self._length = length
        self._tf     = tf
        self._idf    = idf

    def _create_freq_table_sentences(self, sentences):
        sentences_freq_table = {}
        i = 1
        for sentence in sentences:
            words_freq_table = {}
            words = self._pre._lemmatization(sentence)
            words = self._pre._removing_stopwords_in_words_list(words)

            for word in words:
                if word in words_freq_table:
                    words_freq_table[word] += 1
                else:
                    words_freq_table[word] = 1

            doc = 'D' + str(i)
            i += 1
            sentences_freq_table[doc] = words_freq_table #nilai tiap kata dikelompokin berdasarkan kalimatnya

        return sentences_freq_table

    def _create_freq_table_words(self, sentence):
        word_freq_table = {}
        sentence        = self._pre._removing_numbers_in_sentence(sentence)
        sentence        = self._pre._lowering_sentence(sentence)
        sentence        = self._pre._removing_punctuation_in_sentence(sentence)
        words           = self._pre._lemmatization(sentence)
        words           = self._pre._removing_stopwords_in_words_list(words)

        for word in words:
            word_freq_table[word] = word_freq_table.get(word, 0) + 1

        return word_freq_table

    def _calculate_tf_sentences(self, freq_table):
        tf_result = {}

        for doc, sentences_freq_table in freq_table.items():
            tf_table = {}
            #print(sum(sentences_freq_table.values()))
            count_words_in_sentence = sum(sentences_freq_table.values())

            if sentences_freq_table:
                max_count = max(sentences_freq_table.values())
                average_count = count_words_in_sentence/len(sentences_freq_table)
            else:
                continue

            for word, count in sentences_freq_table.items():
                if self._tf == 1: #natural
                    tf_table[word] = count
                if self._tf == 2: #boolean
                    tf_table[word] = 1 if count > 1 else 0
                if self._tf == 3: #logarithm
                    tf_table[word] = 1 + math.log10(count)
                if self._tf == 4: #logarithm 1
                    tf_table[word] = math.log10(1 + count)
                if self._tf == 5: #logarithm average
                    tf_table[word] = (1 + math.log10(count))/(1 + math.log10(average_count))
                if self._tf == 6: #augmented
                    tf_table[word] = 0.5 + (0.5 * count / max_count)
                if self._tf == 7: #inverse tf
                    tf_table[word] = 1 - (1 / ( 1 + count))
                if self._tf == 8: #hans Christian
                    tf_table[word] = count / count_words_in_sentence

            tf_result[doc] = tf_table

        return tf_result

    def _calculate_tf_words(self, sentence):
        word_freq_table = self._create_freq_table_words(sentence)
        tf_result       = {}
        total_word      = len(list(word_freq_table))

        for word, freq in word_freq_table.items():
            tf_result[word] = freq / float(total_word)

        return tf_result

    def _calculate_avg_tf(self, freq1, freq2):
        avg_result = {}
        keys = set(freq1.keys()) | set(freq2.keys())

        for key in keys:
            key1 = freq1.get(key, 0)
            key2 = freq2.get(key, 0)

            avg_result[key] = (key1 + key2) / 2.

        return avg_result

    def _count_per_words(self, freq_table):
        words_freq_table = {}

        for doc, sentences_freq_table in freq_table.items():
            for word, count in sentences_freq_table.items():
                if word in words_freq_table:
                    words_freq_table[word] += 1
                else:
                    words_freq_table[word] = 1

        return words_freq_table

    def _calculate_idf(self, freq_table, count_per_words, total_docs):
        idf_result = {}

        for doc, sentences_freq_table in freq_table.items():
            idf_table = {}

            for word in sentences_freq_table.keys():
                if self._idf == 1: #unary
                    idf_table[word] = 1
                if self._idf == 2: #idf
                    idf_table[word] = math.log10(total_docs / float(count_per_words[word]))
                if self._idf == 3: #prob idf
                    idf_table[word] = numpy.random.uniform(0,(math.log10(total_docs - float(count_per_words[word]) / float(count_per_words[word]))))

            idf_result[doc] = idf_table

        return idf_result

    def _calculate_tf_idf(self, tf_result, idf_result):
        tf_idf_result = {}

        for (doc1, freq_table1), (doc2, freq_table2) in zip(tf_result.items(), idf_result.items()):
            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(freq_table1.items(), freq_table2.items()):
                #TF*IDF
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_result[doc1] = tf_idf_table

        return tf_idf_result

    def _score_sentences(self, tf_idf_result) -> dict:
        sentenceValue = {}

        for doc, f_table in tf_idf_result.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            if(count_words_in_sentence != 0) :
                sentenceValue[doc] = total_score_per_sentence

        return sentenceValue

    def _find_average_score(self, sentenceValue) -> int:
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average

    def sort_func(self,x):
        return x[1:]

    def _create_summary(self, sentences, sentenceValue, length, total_docs):
        threshold = (int)(length * total_docs / 100)
        sentenceValue = (sorted(sentenceValue.items(), key=lambda tup: tup[1], reverse=True))[:threshold]
        summary = ''

        sentenceValue = sorted(sentenceValue, key=lambda tup: int(tup[0][1:]))

        for value in sentenceValue:
            doc = int(value[0][1:])-1
            summary += sentences[doc] + "\n"

        return summary

    def _create_summary2(self, sentences, sentenceValue, length, total_docs):
        summary = ''
        i = 1
        for sentence in sentences:
            doc ='D' + str(i)
            if doc in sentenceValue and sentenceValue[doc] >= (length):
                summary += sentence + '\n'
            i += 1
        return summary

    def _divergence(self, sentences_freq, doc_freq):
        value = 0
        for word, freq in sentences_freq.items():
            if word in doc_freq:
                value += freq * math.log(freq / float(doc_freq[word]))
        return value

    def _js_divergence(self, sentences, doc_freq):
        #menghitung tf untuk kalimat tersebut
        summary_freq = self._calculate_tf_words(' '.join(sentences))
		
        #menghitung tf rata-rata antara kalimat tersebut dengan keseluruhan teks
        average_freq = self._calculate_avg_tf(summary_freq, doc_freq)
		
        kl_summary_average 	= self._divergence(summary_freq, average_freq)
        kl_doc_average 		= self._divergence(doc_freq, average_freq)
	
        jsd 				= kl_summary_average + kl_doc_average
		
        return jsd

    def _tfidf_summarizing(self, sentences):
        # 1 Sentence preprocessing
        sentResult = self._pre._removing_numbers_in_sentence(sentences)
        sentResult = self._pre._lowering_sentence(sentences)
        sentResult = self._pre._sentence_tokenizing(sentResult)
        sentResult = self._pre._removing_whitespaces_in_sentences_list(sentResult)
        sentResult = self._pre._removing_punctuation_in_sentences_list(sentResult)
        total_docs = len(sentResult)

        # 2 Create the Frequency table of the words in each doc/sentence.
        freq_result = self._create_freq_table_sentences(sentResult)

        # 3 creating table total documents per words
        count_per_words = self._count_per_words(freq_result)

        # 4 Calculate TF
        tf_result = self._calculate_tf_sentences(freq_result)
		
        # 5 Calculate IDF
        idf_result = self._calculate_idf(freq_result, count_per_words, total_docs)

        # 6 Calculate TF-IDF
        tf_idf_result = self._calculate_tf_idf(tf_result, idf_result)

        # 7 Important Algorithm: score the sentences
        sentence_scores = self._score_sentences(tf_idf_result)

        # 8 Important Algorithm: Generate the summary
        sentences = self._pre._sentence_tokenizing(sentences)
        summary = self._create_summary(sentences, sentence_scores, self._length, total_docs)

        return summary
