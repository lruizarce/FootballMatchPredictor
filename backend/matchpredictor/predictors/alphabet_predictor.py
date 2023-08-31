# This model chooses the first team alphabetically to win

from typing import List, Tuple, Optional, cast


from matchpredictor.matchresults.result import Fixture, Outcome, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

class AlphabetPredictor(Predictor):
    def predict(self, fixture: Fixture) -> Prediction:
        if fixture.home_team.name < fixture.away_team.name:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.AWAY)