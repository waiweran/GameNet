import socket	
import _thread		 
import json
import gain


try:
	
	# Initialize Network
	model = gain.GainModel()

	# Create a socket 
	s = socket.socket()		 
	port = 12345				
	s.bind(('', port))		 
	s.listen(5)	 
	print('socket is listening on port {port}'.format(port=port))			

	# Accept connections and run moves
	while True:
		conn, addr = s.accept()	 
		instring = connection.recv(2048).decode()
		if instring.startswith('{'):
			output = model.run_game(instring)
			connection.send((output + '\n').encode())
		conn.close()

finally: 
	s.close()
