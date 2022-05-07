from copy import deepcopy
import operator

class PlayerPool:
    def __init__(self, elo):
        self.elo = elo
        self.pool = []
        self.counter = 0
        
    def add_player(self, model):
        model = deepcopy(model)
        self.pool.append(model)
        
        self.elo.addPlayer(self.counter, 
                           rating=1000 if self.counter <= 1 else self.elo.getRating(self.counter - 1)[0])
        self.counter += 1
        
    def get_latest_player_model(self):
        return self.pool[-1]
    
    def get_latest_player_rating(self):
        return self.elo.getRating(self.counter - 1)[0]
    
    def update_elo_rating(self, winner_id, loser_id):
        self.elo.updateRating(winner_id, loser_id)

    def save_best_player(self):
        best_rating = max(self.elo.ratingDict.items(), key=operator.itemgetter(1))[0]
        best_model = self.pool[best_rating]
        best_model.save_checkpoint('.', filename='best_model')
        