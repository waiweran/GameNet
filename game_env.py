import subprocess
import socket
import json

socket.setdefaulttimeout(1.)

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
        try:
            return json.loads(self.conn.recv(2048).decode())['GainChoice']
        except socket.timeout:
            return self.reset()


    def step(self, move):
        if move == 'random':
            move = [move]
        else:
            move = move.tolist()
        self.conn.send((json.dumps(move + '\n')).encode())
        instring = ""
        count = 0
        while not instring.startswith('{'):
            try:
                instring = self.conn.recv(2048).decode()
            except socket.timeout:
                return [], 0, True, 0, 'Game Failure'
            count += 1
            if count > 100:
                return [], 0, True, 0, 'Game Failure'
        indict = json.loads(instring)
        return indict['GainChoice'], indict['Reward'], indict['Done'], indict['Action'], indict['Score']


    def close(self):
        if self.sim and not self.sim.poll():
            self.sim.kill()
        if self.conn:
            self.conn.close()
        self.soc.close()
