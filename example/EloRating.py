class Elo:
    def __init__(self, k=16):
        self.ratingDict = {}
        self.k = k
    
    def getRating(self, name):
        return self.ratingDict[name]
    
    def setRating(self, name, rating):
        self.ratingDict[name][0] = rating
        self.ratingDict[name][1] += 1

    def addPlayer(self, name, rating=1000):
        self.ratingDict[name] = [rating, 0]
        
    def updateRating(self, winner, loser):
        EA = self.expectedResult(winner, loser)
        EB = 1 - EA
        
        RA, CA = self.getRating(winner)
        RB, CB = self.getRating(loser)
        
        if CA >= 20 or CB >=20:
            self.k = 32
        else:
            self.k = 16
        
        self.setRating(winner, RA + self.k * (1 - EA))
        self.setRating(loser, RB + self.k * (0 - EB))
    
    def expectedResult(self, playerA, playerB):
        RA, _ = self.getRating(playerA)
        RB, _ = self.getRating(playerB)

        exp = (RB - RA) / 400.0
        return 1 / (1 + 10 ** exp)
    
    def exhibitAllScores(self):
        print(self.ratingDict)
    