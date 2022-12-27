import sys

class Logger:
    def __init__(self,log_name):
        self.log_name = log_name
        self.terminal = sys.stdout
        self.log = open(log_name, "w")
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.log_name, "a")
        self.log.write(message)
        self.log.close() #to make log readable at all times...

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass