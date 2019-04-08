import subprocess
import socket
import json

class Dominion:

    def __init__(self, opponent='Big Money'):
        self.opponent = opponent
        self.sim = None
        self.conn = None

        # Create a socket 
        self.soc = socket.socket()      
        port = 12345                
        self.soc.bind(('', port))       
        self.soc.listen(5)


    def reset(self):

        # Get rid of existing simulator
        if self.sim and not self.sim.poll():
            self.sim.kill()
        if self.conn:
            self.conn.close()

        # Start simulator
        self.sim = subprocess.Popen(['java', '-jar', 'Simulator.jar', '1', 'Socket', self.opponent], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # Connect to game
        self.conn, _ = self.soc.accept()

        # Return first move
        return json.loads(self.conn.recv(2048).decode())['GainChoice']


    def step(self, move):
        self.conn.send((json.dumps(move.tolist()) + '\n').encode())
        instring = ""
        while not instring.startswith('{'):
            try:
                self.conn.settimeout(1)
                instring = self.conn.recv(2048).decode()
            except socket.timeout:
                self.conn.send('{"Resend": true}\n'.encode())
        indict = json.loads(instring)
        return indict['GainChoice'], indict['Reward'], indict['Done'], indict['Action'], indict['Score']


    def close(self):
        if self.sim and not self.sim.poll():
            self.sim.kill()
        if self.conn:
            self.conn.close()
        self.soc.close()
