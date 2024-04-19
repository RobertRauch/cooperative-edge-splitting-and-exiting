class Strategy:
    def __init__(self):
        self.split = 0
        self.autoencoder = -1
        self.thresholds = {"EE 1": 0.9, "EE 2": 0.9, "EE 3": 0.9}
        self.ignore_ee = False
        self.exit = -1

    def update_strategy(self, split, ae, thresholds, ignore_ee=False, e=-1):
        self.split = split
        self.autoencoder = ae
        self.thresholds = thresholds
        self.ignore_ee = ignore_ee
        self.exit = e


class Requirements:
    def __init__(self, latency, accuracy = -1):
        self.latency = latency
        self.accuracy = accuracy

    def update_requirements(self, latency, accuracy = -1):
        self.latency = latency
        self.accuracy = accuracy
