import socket	
import _thread		 
import json
import gain

def run_game(model, connection):
	total = 0
	while(True):
		instring = connection.recv(2048).decode()
		if instring.startswith('{'):
			print("Received " + str(total))
			total += 1
			output = gain.run_game(model, instring)
			connection.send(output + '\n').encode()


try:

	# Initialize Network
	model = gain.load_model()

	# Create a socket 
	s = socket.socket()		 
	port = 12345				
	s.bind(('', port))		 
	s.listen(5)	 
	print('socket is listening on port {port}'.format(port=port))			

	# Accept connections
	while True:
		conn, addr = s.accept()	 
		print('Got connection from {addr}'.format(addr=addr))
		_thread.start_new_thread(run_game, (model, conn))

finally: 
	s.close()