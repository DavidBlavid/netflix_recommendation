class EarlyStopper:

    def __init__(self, patience: int):
        self.patience_left = patience
        self.patience = patience
        self.best_score = 0

    def should_stop(self, top_n_score: float) -> bool:
        self.patience_left -= 1
        if self.patience_left == 0:
            return True
        if self.is_better(top_n_score):
            self.best_score = top_n_score
            self.patience_left = self.patience
        return False
    
    def is_better(self, top_n_score: float) -> bool:
        return self.best_score < top_n_score