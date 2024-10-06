from tkinter			import filedialog
from tkinter			import *
from tkinter			import messagebox
from tkinter.ttk		import *

from audioToText		import AudioToText
from preProcessing		import PreProcessing
from tfidf				import TFIDF
from beeHive			import BeeHive
from threading			import Thread

import os
import time 

if __name__ == '__main__':
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"				#lokasi key buat akses bucket

	output_filepath			= "transcripts/"								#Final transcript path
	bucket_name				= "meeting_summarizer_data"							#nama bucket buat naro file-nya
	punctuation				= True 											#kalo mau pake punctuation di hasil audio to text
	delete_from_blob		= False 										#apakah audio mau dihapus dari bucket
	tfidf_combine_ln		= 70											#berapa persen dari total panjang summary dari TFIDF untuk diproses lagi sama ABCO
	abco_len				= 20											#maksimum karakter pada hasil ringkasan dari proses ABCO
	max_epoch				= 30											#maksimum skor total fitness (batas usaha lebah mencari makanan baru)
	max_trial				= 10											#maksimum lebah mencoba (batas makanan diabaikan / tidak menemukan yang lebih baik lagi)
	maximization			= False 										#positif atau negatif
	colony_size				= 100											#jumlah lokasi makanan = jumlah lebah
	result					= ""
	summary					= ""
	filename				= ""
	window					= Tk()
	progress				= Progressbar(window, orient = HORIZONTAL, length = 800, mode = 'determinate')
	progress['value'] 		= 0
	window.resizable(0, 0) # In order to prevent the window from getting resized you will call 'resizable' method on the window
	window.title("Meeting Summarizer")
	
	def bar(thread):
		thread.start()
		progress.grid(row=5, column = 1, columnspan=2, padx=5, pady = 10) 
		
		value = 0
		
		while thread.is_alive():
			if value < 25:
				time.sleep(0.5)
			if value >= 25 and value < 50:
				time.sleep(1)
			if value >= 50 and value < 75:
				time.sleep(1.25)
			if value >= 75 and value < 85:
				time.sleep(1.5)
			if value >= 85 and value < 98:
				time.sleep(1.75)
			
			if value < 98:
				value += 0.25
				setvalue(value)
				
			window.update()
			pass
		
		setvalue(100)
		window.update()
		
		return summary
		
	def temporary_aja():
		value = 0
		flag = 'up'
		while True:
			progress['value'] = value
			window.update_idletasks() 
			time.sleep(0.5) 
			if value == 100:
				flag = 'down'
			if value == 0:
				flag = 'up'
			
			if flag == 'up':
				value += 20
			if flag == 'down':
				value -= 20
				
	def setvalue(value):
		progress['value'] = value
	
	def summarize():
		global filename
		temp1			= filename.split('/')
		audio_file_name	= temp1[len(temp1)-1]
		temp2			= len(filename)-len(audio_file_name)
		filename		= filename[:temp2]
		
		audio_to_text	= AudioToText(filename, audio_file_name, output_filepath, bucket_name, punctuation)		#ini untuk upload audio
		transkrip		= audio_to_text._convert_audio_to_text(delete_from_blob)								#ini untuk convert audio ke teks
		length			= len(PreProcessing()._sentence_tokenizing(transkrip))
		abco_ln			= (int)(length*abco_len/100)
		result			= TFIDF(tfidf_combine_ln)._tfidf_summarizing(transkrip) #hasil dari TFIDF
		beeHive 		= BeeHive(
						   sentence			= result,
						   food_capacity	= abco_ln,
						   max_epoch		= max_epoch,
						   max_trial		= max_trial,
						   maximization		= maximization,
						   colony_size		= colony_size,
						   )
		summary 		= beeHive._run()
		outputSummarize.insert('1.0',summary)
	
	def btn_upload():
		global filename
		filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype = (("all files","*.*"),("jpeg files","*.jpg")))
		outputUpload.insert('1.0', filename) #required 2 arguments

	def btn_summarize():
		try:
			thread1 = Thread(target=summarize, args=())
			result = bar(thread1)
		except:
			print("Error: unable to start thread")

	def btn_save():
		savefilename = filedialog.asksaveasfilename(defaultextension="txt",filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
		if not savefilename:
			return
		with open(savefilename, "w") as output_file:
			text = outputSummarize.get("1.0", END)
			output_file.write(text)
		os.startfile(savefilename) 
	
    #Creating the widgets
	upload_btn		= Button(window, text="Upload Meeting Audio", width=20)
	summarize_btn	= Button(window, text="Summarize", width=20)
	save_btn		= Button(window, text="Save As..", width=20)
	labelUpload		= Label(window, text="Audio Filepath:")
	labelSummarize	= Label(window, text="Summarized Meeting:")
	outputUpload	= Text(window, width=100, height=2, wrap=WORD)
	outputSummarize	= Text(window, width=100, height=20, wrap=WORD)

    #Positioning the widgets
	upload_btn.grid(row=1, column=1, columnspan=2, pady=5)
	labelUpload.grid(row=2, column=1, padx=5, sticky=W)
	outputUpload.grid(row=3, column=1, columnspan=2, padx=5, pady=(0,10))
	summarize_btn.grid(row=4, column=1, columnspan=2, pady=5)
	labelSummarize.grid(row=6, column=1, padx=5, sticky=W)
	outputSummarize.grid(row=7, column=1, columnspan=2, padx=5, pady=(0,10))
	save_btn.grid(row=8, column=1, columnspan=2, pady=5)

    #Button activation
	upload_btn.configure(command=btn_upload)
	summarize_btn.configure(command=btn_summarize)
	save_btn.configure(command=btn_save)

	window.mainloop()