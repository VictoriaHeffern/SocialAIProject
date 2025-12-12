import socket
import threading
import json
import time
import sys

try:
    from naoqi import ALProxy, ALModule # See if the naoqi is accessible 
except ImportError:
    print ("naoqi not found.")
    class MockTTS:
        def say(self, text):
            print ("[Mock NAO] NAO would say: %s" % text)
    class MockALProxy:
        def __init__(self, *args):
            if args[0] == "ALTextToSpeech": self.proxy = MockTTS()
        def say(self, text):
            if self.proxy: self.proxy.say(text)
    ALProxy = MockALProxy

####################################################################################################

class NaoSpeechServer:
    def __init__(self, nao_ip, host='0.0.0.0', port=9000):
        self.host = host
        self.port = port
        self.nao_ip = nao_ip
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_conn = None
        self.running = False
        self.tts = None
        self.app = None
        self.memory = None

        try:
            print ("Connecting to NAO AnimatedSpeech at %s:9559..." % nao_ip)
            import qi
            self.app = qi.Application(["NaoSpeechServer", "--qi-url=tcp://{}:9559".format(self.nao_ip)])
            self.app.start()

            self.tts = self.app.session.service("ALAnimatedSpeech") # Setup tts as animated
            self.memory = self.app.session.service("ALMemory")
            print ("NAOqi AnimatedSpeech connected successfully.")

        except Exception as e:
            print ("Failed to connect to NAO AnimatedSpeech: %s" % e)
            print ("Check IP and network connection.")

    # start() method is responsible for init of the NAO server
    def start(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            print ("Server listening on %s:%s..." % (self.host, self.port))
            threading.Thread(target=self.accept_connections, args=(), name='AcceptThread').start()
        except Exception as e:
            print ("Error starting server: %s" % e)

    # accept_connections() method
    def accept_connections(self):
        while self.running:
            try:
                self.client_conn, addr = self.server_socket.accept() # accept() blocks until a client connects
                print ("Client connected from %s" % str(addr))
                
                self.client_conn.send(json.dumps({'status': 'ok'})) # Send confirmation
                
                threading.Thread(target=self.handle_client, args=(self.client_conn,), name='ClientThread').start() # Handle client in new thread
            except Exception:
                if self.running: print ("Connection error in accept loop.")
                break

    # handle_client() method
    def handle_client(self, conn):
        while self.running:
            try:
                data = conn.recv(4096) # did we recieve anything?
                if not data: break
                
                message = json.loads(data)
                
                if 'speech' in message and 'response' in message['speech']:
                    text_to_say = message['speech']['response']
                    print ("\n[NAO SPEECH] --> %s" % text_to_say)
                    if self.tts:
                        animated_text = "^mode(contextual)"+text_to_say+"^mode(contextual)" # tag the text with contextual body language
                        self.tts.say(animated_text) # send to NAO tts

                elif 'command' in message and message['command'] == 'stop':
                    print ("Received stop command.")
                    break
                    
            except Exception:
                break
        
        if conn == self.client_conn:
            self.client_conn.close()
            self.client_conn = None
            print ("Client disconnected.")

    # stop() method
    def stop(self):
        self.running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
        try:
            self.server_socket.close()
        except:
            pass
        if self.app:
            self.app.stop()
        print("Server shut down.")
        sys.exit(0)


####################################################################################################

if __name__ == "__main__":
    NAO_IP = "192.168.0.172" 
   
    server = NaoSpeechServer(nao_ip=NAO_IP)
    server.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print ("\nStopping server...")
    finally:
        server.stop()