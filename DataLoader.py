
class DataLoader():
    def __init__(self, filename, device):
        self.filename = filename
        self.device = device

    def caseload(self):
        nl = []
        description = ""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if  line != "\n":
                    description += line
                else:
                    nl.append(description)
                    description = ""
        return nl