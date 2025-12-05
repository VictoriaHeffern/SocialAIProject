import socket
import json
import threading
import time
import queue
import sounddevice as sd
import numpy as np
import os
import wave
from tempfile import NamedTemporaryFile

from AIConvoModel import AIConvoModel 

charPerSec = 13.5

class ConversationClient:
    def __init__(self, server_host, server_port, face_known=False, person_name=None):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.connected = False
        self.listening = False
        self.listen_lock = threading.Lock()

        self.ai_model = AIConvoModel(face_known=face_known, person_name=person_name)

        self.q = queue.Queue()
        self.samplerate = 16000
        self.channels = 1
        self.dtype = 'int16'

        self.speech_threshold = 1500
        self.silence_threshold = 800
        self.silence_duration = 1.5
        self.max_silence_listen_duration = 5.0
        self.max_listen_duration = 20.0

        print("Conversation client initialized.")

    def connect(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_host, self.server_port))

            init_message = {'type': 'conversation', 'version': '1.0'}
            self.client_socket.send(json.dumps(init_message).encode())

            response = json.loads(self.client_socket.recv(1024))

            if response.get('status') == 'ok':
                self.connected = True
                print("Connected to local server.")
                return True
            else:
                print(f"Failed to register with server: {response.get('message')}")
                self.client_socket.close()
                return False

        except Exception as e:
            print(f"Connection error: {e}")
            if self.client_socket:
                self.client_socket.close()
            return False

    def audio_callback(self, indata, frames, time_, status):
        if status:
            print(f"Audio Status: {status}", flush=True)
        self.q.put(bytes(indata))

    def recognize_speech(self):
        with NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name

        all_audio_data = []
        recording_started = False
        start_listen_time = time.time()
        continuous_silence_start = None

        print("\nWaiting for user dialogue...")

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, dtype=self.dtype,
                               channels=self.channels, callback=self.audio_callback):
            while self.listening:
                if time.time() - start_listen_time > self.max_silence_listen_duration and not recording_started:
                    print("No speech detected within timeout.")
                    break
                elif time.time() - start_listen_time > self.max_listen_duration:
                    print("Max listen duration reached.")
                    break

                try:
                    data = self.q.get(timeout=0.1)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                    if rms > self.speech_threshold and not recording_started:
                        recording_started = True
                        continuous_silence_start = None
                        print(f"Speech detected! RMS: {rms:.2f} - Recording started")

                    if recording_started:
                        all_audio_data.append(data)

                        if rms > self.silence_threshold:
                            continuous_silence_start = None
                        elif continuous_silence_start is None:
                            continuous_silence_start = time.time()
                        elif time.time() - continuous_silence_start >= self.silence_duration:
                            print(f"Silence detected for {self.silence_duration}s - stopping recording.")
                            break

                except queue.Empty:
                    continue

        if not recording_started or not all_audio_data:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            return None

        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(b''.join(all_audio_data))

            # Use AIConvoModel's client to call whisper transcription:
            with open(temp_filename, "rb") as audio_file:
                response = self.ai_model.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
                transcription = response.text.strip()
                print(f"User said: **{transcription}**")
                return transcription

        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def listen_for_input(self):
        initial_nao_prompt = self.ai_model.get_initial_greeting()
        print(f"\nNAO START: {initial_nao_prompt}")
        self.send_speech({"response": initial_nao_prompt})

        initmsglen = len(initial_nao_prompt)
        initmsgtime = initmsglen / charPerSec
        time.sleep(initmsgtime)
        print("Init wait time for: ", initmsgtime)

        while self.listening and True:
            user_input = self.recognize_speech()

            with self.listen_lock:
                if not self.listening:
                    print("Listening stopped by flag.")
                    break

                user_input_for_gpt = user_input if user_input else "The user was quiet and did not answer the question."

                # Use AIConvoModel for GPT response
                response_text = self.ai_model.ask_gpt(user_input_for_gpt)
                response = {"response": response_text}

                print(f"NAO REPLY: {response['response']}")
                self.send_speech(response)

                msglen = len(response['response'])
                msgtime = msglen / charPerSec
                time.sleep(msgtime)

        if self.connected:
            self.client_socket.send(json.dumps({'command': 'stop'}).encode())
        self.listening = False
        print("\n--- Conversation Test Complete ---")

    def start_listening(self):
        with self.listen_lock:
            if not self.listening:
                self.listening = True
                self.listen_thread = threading.Thread(target=self.listen_for_input)
                self.listen_thread.daemon = True
                self.listen_thread.start()
                print("Starting conversation...")

    def send_speech(self, text_dict):
        if not self.connected:
            print("Not connected to server. Cannot send speech.")
            return False
        try:
            message = {'speech': text_dict}
            self.client_socket.send(json.dumps(message).encode())
            return True
        except Exception as e:
            print(f"Error sending speech: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        self.connected = False
        print("Disconnected from server.")
