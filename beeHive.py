from preProcessing		import PreProcessing
from copy				import deepcopy
from tfidf 				import TFIDF
import numpy

class BeeHive(object):
	def __init__(self, sentence, food_capacity, max_epoch, max_trial, maximization, colony_size):
		self._pre					= PreProcessing()																		#inisialisasi preprocessing
		self._word_freq		  		= TFIDF()._calculate_tf_words(sentence)													#frequency dari setiap kata
		self._colony_size			= colony_size 																		#banyak populasi lebah
		self._foods		  			= self._pre._sentence_tokenizing(sentence)												#sekumpulan kalimat yang menjadi kandidat lokasi makanan
		self._food_capacity			= food_capacity																			#batas makanan yang dapat diambil
		self._fitness_func			= TFIDF()._js_divergence 																#function untuk menghitung skor dari lokasi makanan
		self._foods_locations 		= self._initial_foods_location()														#lokasi makanan inisialisasi
		self._foods_locations_score = [self._calculate_food_score(location) for location in self._foods_locations] 			#skor dari masing2 lokasi makanan inisialisasi
		self._trials				= [0] * colony_size 																#data yg berisi brp kali lebah tidak menemukan lokasi makanan lain yang lebih baik
		self._fitness_evaluation	= colony_size 																		#berapa kali lebah mencari lokasi makanan baru
		self._epoch	 				= 0
		self._max_epoch				= max_epoch																				#batas epoch
		self._best_location 		= (None, -10000) if maximization else (None, 10000)										#lokasi makanan terbaik
		self._probabilities			= []																					#data yg berisi probabilitas dari setiap lokasi makanan untuk dipilih

		#inisialisasi lebah
		self._employedBee			= EmployedBee(maximization, max_trial, food_capacity, self._foods, self._word_freq, self._fitness_func)
		self._onlookerBee			= OnlookerBee(maximization, max_trial, food_capacity, self._foods, self._word_freq, self._fitness_func)
		self._scoutBee 				= ScoutBee(maximization, max_trial, food_capacity, self._foods, self._word_freq, self._fitness_func)

	def _run(self):
		foods  = self._find_best_food()
		idx    = []
		result = ''
		
		#posisi kalimat dalam lokasi makanan tidak berurutan, sehingga tahap dibawah ini adalah untuk mengurutkan kalimat sesuai dengan urutan teks aslinya
		for food in foods[0]:
			idx.append(self._foods.index(food))

		idx = sorted(idx)

		for i in idx:
			result += self._foods[i] + '\n'

		return result

	def _find_best_food(self):
		while True:
			self._epoch += 1
			#jika banyak epoch sudah memenuhi batas dari max_epoch maka lokasi terbaik terakhir akan diambil
			if self._epoch > self._max_epoch:
				return self._best_location

			print("Epoch : ", self._epoch, " | Fitness Evaluation : ", self._fitness_evaluation, " | Best Location : ", self._best_location[1])

			#Employe Bee (Lebah Pekerja)
			self._foods_locations, self._foods_locations_score, self._best_location, self._fitness_evaluation, self._trials, self._probabilities = self._employedBee._finding_foods(self._foods_locations, self._foods_locations_score, self._best_location, self._fitness_evaluation, self._trials)

			#Onlooker Bee (Lebah Pengamat)
			self._foods_locations, self._foods_locations_score, self._best_location, self._fitness_evaluation, self._trials = self._onlookerBee._onlooking_foods(self._colony_size, self._probabilities, self._foods_locations, self._foods_locations_score, self._best_location, self._fitness_evaluation, self._trials)

			#Scout Bee (Lebah Pengintai)
			self._best_location, self._fitness_evaluation, self._trials = self._scoutBee._scouting_foods(self._best_location, self._fitness_evaluation, self._trials)

	def _initial_foods_location(self):
		foods_locations = []
		#bikin lokasi makanan sebanyak populasi yang sudah ditentukan
		for i in range(self._colony_size):
			location = self._create_foods_locations()
			foods_locations.append(location)
		return foods_locations

	def _create_foods_locations(self):
		#bikin score random untuk masing2 kalimat
		random_scores 	 = numpy.random.rand(len(self._foods))

		#kalimat ama score-nya dipasang2in secara random
		scored_sentences = zip(self._foods, random_scores)

		#diurutin dari score paling gede, lambda tup: tip[1] maksudnya diurutin berdasarkan nilai array ke 1 dari list itu
		sorted_sentences = sorted(scored_sentences, key=lambda tup: tup[1], reverse=True)

		return self._chooser(sorted_sentences)
	
	#untuk memilih beberapa kalimat dari sekumpulan kalimat yang ada
	def _chooser(self, sentences):
		chosen 	    = []
		count 		= 0
		for i in range(len(sentences)):
			if count < self._food_capacity:
				chosen.append(sentences[i][0])
				del sentences[i]
				count+=1
			else:
				break

		return chosen

	#untuk hitung skor dari summary
	def _calculate_food_score(self, food_location):
		return self._fitness_func(food_location, self._word_freq)

class Bee(object):
	def __init__(self, maximization, max_trial, food_capacity, foods, doc_freq, fitness_func):
		self._maximization  	= maximization
		self._max_trial			= max_trial
		self._food_capacity 	= food_capacity  	#maximum length / total word for summary
		self._foods				= foods 			#kumpulan kalimat
		self._word_freq			= doc_freq 			#ganti nama apa yah ???
		self._fitness_func 		= fitness_func

	#lokasi / summary mana yang lebih baik
	def _which_better(self, a, b):
		if self._maximization:
			return a > b
		return a < b

	#cari kemungkinan komposisi summary lain yang lebih baik secara random
	def _find_other_locations(self, old_food_location):
		food_location = deepcopy(old_food_location)
		idx 		  = numpy.random.randint(low=0,high=len(food_location))
		del food_location[idx]

		while True:
			idx = numpy.random.randint(low=0,high=len(food_location))
			if self._foods[idx] not in food_location:
				food_location.append(self._foods[idx])
				break

		return food_location

	#untuk hitung skor dari summary
	def _calculate_food_score(self, food_location):
		return self._fitness_func(food_location, self._word_freq)

class EmployedBee(Bee):
	def _finding_foods(self, foods_locations, foods_locations_score, best_location, fitness_evaluation, trials):
		for i, location in enumerate(foods_locations):
			new_location 		= self._find_other_locations(location) 		#cari lokasi baru
			new_score	 		= self._calculate_food_score(new_location)	#hitung skor lokasi baru
			fitness_evaluation += 1

			if self._which_better(new_score, foods_locations_score[i]):		#jika skor yang baru lebih baik dari yg sekarang
				if self._which_better(new_score, best_location[1]): 		#jika skor yang baru lebih baik dari best location
					best_location = (new_location, new_score)

				foods_locations_score[i]    = new_score
				foods_locations[i] 			= new_location
				trials[i] = 0
			else:
				trials[i] += 1

		sum_vector 	  = sum(foods_locations_score)
		probabilities = [score / float(sum_vector) for score in foods_locations_score] #skor dari masing-masing makanan diubah ke persenan

		return foods_locations, foods_locations_score, best_location, fitness_evaluation, trials, probabilities

class OnlookerBee(Bee):
	def _onlooking_foods(self, colony_size, probabilities, foods_locations, foods_locations_score, best_location, fitness_evaluation, trials):
		s = 0
		t = 0
		while t < colony_size:
			probability = numpy.random.uniform(0,1)
			if probability < probabilities[s]: #jika probabilitas-nya lebih kecil dari yg udah ada
				t 			+= 1
				new_location = self._find_other_locations(foods_locations[s])	#cari lokasi baru
				new_score 	 = self._calculate_food_score(new_location) 		#hitung skor lokasi baru
				fitness_evaluation += 1

				if self._which_better(new_score, foods_locations_score[s]):		#jika skor yang baru lebih baik dari yg sekarang
					if self._which_better(new_score, best_location[1]):			#jika skor yang baru lebih baik dari best location
						best_location = (new_location, new_score)

					foods_locations_score[s] = new_score
					foods_locations[s]       = new_location
					trials[s] = 0
				else:
					trials[s] += 1

			s += 1
			if s == colony_size:
				s = 0

		return foods_locations, foods_locations_score, best_location, fitness_evaluation, trials

class ScoutBee(Bee):
	def _scouting_foods(self, best_location, fitness_evaluation, trials):
		max_idx = trials.index(max(trials))
		if trials[max_idx] > self._max_trial: #jika trial sudah lebih besar dari batas yg sudah ditentukan
			new_location = self._find_other_locations() 			#bikin lokasi makanan baru
			new_score 	 = self._calculate_food_score(new_location) #hitung skor lokasi makanan baru
			fitness_evaluation += 1
			trials[max_idx] = 0

			if self._which_better(new_score, best_location[1]):
				best_location = (new_location, new_score)

		return best_location, fitness_evaluation, trials

	def _find_other_locations(self):
		random_scores 	 = numpy.random.rand(len(self._foods))
		scored_sentences = zip(self._foods, random_scores)
		sorted_sentences = sorted(scored_sentences, key=lambda tup: tup[1], reverse=True)
		return self._chooser(sorted_sentences)

	def _chooser(self, sentences):
		chosen 	    = []
		count 		= 0
		for i in range(len(sentences)):
			if count < self._food_capacity:
				chosen.append(sentences[i][0])
				del sentences[i]
				count+=1
			else:
				break
		return chosen
