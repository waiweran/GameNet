import subprocess
import socket
import json

socket.setdefaulttimeout(5.)

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
        self.sim = subprocess.Popen(['java', '-jar', 'Simulator.jar', '-q', '1', 'Stdio', self.opponent], 
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Return first move
        return json.loads(self.sim.stdout.readline().decode())['GainChoice']


    def step(self, move):
        try:
            move = move.tolist()
        except AttributeError:
            pass
        self.sim.stdin.write((json.dumps(move) + '\n').encode())
        self.sim.stdin.flush()
        indict = json.loads(self.sim.stdout.readline().decode())
        return indict['GainChoice'], indict['Reward'], indict['Done'], indict['Action'], indict['Score']


    def close(self):
        if self.sim:
            self.sim.kill()
