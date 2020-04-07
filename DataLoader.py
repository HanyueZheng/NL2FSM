
class DataLoader():
    def __init__(self, filename, targetfile, device):
        self.filename = filename
        self.targetfile = targetfile
        self.device = device

    def caseload(self):
        nl = []
        target = []
        description = ""
        with open(self.targetfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if  line != "\n":
                    description += line
                else:
                    target.append(description)
                    description = ""

        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if  line != "\n":
                    description += line
                else:
                    nl.append(description)
                    description = ""
        return nl, target
