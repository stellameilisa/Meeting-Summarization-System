from google.cloud           import storage
from pydub                  import AudioSegment
from google.cloud           import speech
from google.cloud.speech    import enums
from google.cloud.speech    import types
import os

class AudioToText(object):
    #inisialisasi ketika object dibuat
    def __init__(self, filepath, audio_file_name, output_filepath, bucket_name, punctuation):
        self._filepath          = filepath															#path folder letak file
        self._audio_file_name   = audio_file_name													#nama file
        self._output_filepath   = output_filepath													#path transkrip output hasil convert
        self._bucket_name       = bucket_name														#nama bucket untuk upload file
        self._is_punctuation    = punctuation														#apakah pakai auto punctuation ?
        self._is_exists         = False																#apakah filenya ada ?

    #untuk meng-upload file ke google cloud storage
    def _upload_blob(self, bucket_name, source_file_name, audio_file_name, destination_blob_name):
        storage_client  = storage.Client()															#memanggil storage
        bucket          = storage_client.bucket(bucket_name)										#membuat bucket dalam storage jika belom ada
        blob            = bucket.blob(destination_blob_name)										#membuat blob dalam bucket jika belom ada

        stats = storage.Blob(bucket=bucket, name=audio_file_name).exists(storage_client)			#apakah filenya sudah ada di bucket ?

        if not stats:
            blob.upload_from_filename(source_file_name)												#jika belom ada maka audio di-upload
            print("File {} uploaded to {}.".format(audio_file_name, bucket_name))
        else:
            print("File {} exists in {}".format(audio_file_name, bucket_name))
            self._is_exists = True																	#file sudah ada

    #untuk menghapus file yang di-upload di google cloud storage
    def _delete_blob(self, bucket_name, blob_name):
        storage_client  = storage.Client()															#memanggil storage
        bucket          = storage_client.get_bucket(bucket_name)									#memanggil bucket dalam storage
        blob            = bucket.blob(blob_name)													#memanggil blob dalam storage
        blob.delete()																				#menghapus file

    #untuk mengetahui frame rate dari audio yg di-upload
    def _get_frame_rate(self, audio_file_name):
        return AudioSegment.from_mp3(audio_file_name).frame_rate

    #untuk recognition audio-nya
    def _google_transcribe(self, audio_file_name, delete):
        file_path               = self._filepath + audio_file_name									#path file
        print("Path File : ", file_path)
        
        frame_rate              = self._get_frame_rate(file_path)									#frame rate
        print("File Frame Rate : ", frame_rate)
        
        transcript              =  ''																#variable penampung trankrip hasil convert
        destination_blob_name   = audio_file_name													#nama file yang akan di-upload ke google cloud storage
        output_filepath         = self._output_filepath + audio_file_name.split('.')[0] + '.txt'	#nama transkrip

        if os.path.isfile(output_filepath):
            file        = open(output_filepath,'r')													#jika transkrip sudah ada
            transcript  = file.read()																#baca transkrip
        else:
            print("\nRecognize the audio")
            self._upload_blob(self._bucket_name, file_path, audio_file_name, destination_blob_name)	#jika trankrip belom ada maka upload file
            gcs_uri     = 'gs://' + self._bucket_name + '/' + audio_file_name						#path folder di google cloud storage

            client      = speech.SpeechClient()														#memanggil storage
            audio       = types.RecognitionAudio(uri = gcs_uri)										#mengecek tipe audio
            encoding    = enums.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED				#encoding tidak diketahui
            language    = 'en-US'																	#bahasa yang digunakan dalam audio

            config      = types.RecognitionConfig(													#config
                           encoding                     = encoding,
                           sample_rate_hertz            = frame_rate,
                           language_code                = language,
                           enable_automatic_punctuation = self._is_punctuation)

            operation   = client.long_running_recognize(config, audio)								#recognize audio
            response    = operation.result(timeout = 10000)											#timeout jika tidak ada response

            for result in response.results:
                transcript += result.alternatives[0].transcript										#memasukkan response kedalam transkrip

            if delete:
                delete_blob(bucket_name, destination_blob_name)										#jika ingin dihapus maka akan dihapus
                print(audio_file_name, ' deleted from bucket!')

        return transcript

    #menyimpan transkrip
    def _write_transcripts(self, transcript_filename, transcript):
        f = open(self._output_filepath + transcript_filename,"w+")
        f.write(transcript)
        f.close()

    #convert audio
    def _convert_audio_to_text(self, delete):
        transcript          = self._google_transcribe(self._audio_file_name, delete)
        transcript_filename = self._audio_file_name.split('.')[0] + '.txt'							#nama trankrip

        self._write_transcripts(transcript_filename,transcript)
        print("\nTrancribe succeed!")
        return transcript
