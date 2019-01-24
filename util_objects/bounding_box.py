class Box_Predictor:
    def __init__(self, label, score, y1, x1, y2, x2):
        self.label = label
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # def set_label(self,label):
    #     self.label = label
    #
    # def set_label(self,score):
    #     self.score = score

