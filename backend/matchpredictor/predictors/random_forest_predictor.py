from typing import List, Tuple, Optional
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction


class RandomForestPredictor(Predictor):
    def __init__(self, model: RandomForestRegressor, team_encoding: OneHotEncoder) -> None:
        self.model = model
        self.team_encoding = team_encoding
    
    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)
        
        if encoded_home_name is None or encoded_away_name is None:
            return Prediction(outcome=Outcome.DRAW)
        
        x: NDArray[float64] = np.concatenate([encoded_home_name, encoded_away_name], 1)
        pred = self.model.predict(x)

        if pred > 0:
            return Prediction(outcome=Outcome.HOME)
        elif pred < 0:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)
        
    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        try:
            result: NDArray[float64] = self.team_encoding.transform(np.array(team.name).reshape(-1, 1))
            return result
        except ValueError:
            return None
        
    @classmethod
    def build_model(cls, results: List[Result]) -> Tuple[RandomForestRegressor, OneHotEncoder]:
        home_names = np.array([r.fixture.home_team.name for r in results])
        away_names = np.array([r.fixture.away_team.name for r in results])
        home_goals = np.array([r.home_goals for r in results])
        away_goals = np.array([r.away_goals for r in results])

        team_names = np.array(list(home_names) + list(away_names)).reshape(-1, 1)
        team_encoding = OneHotEncoder(sparse=False).fit(team_names)

        encoded_home_names = team_encoding.transform(home_names.reshape(-1, 1))
        encoded_away_names = team_encoding.transform(away_names.reshape(-1, 1))

        x: NDArray[float64] = np.concatenate([encoded_home_names, encoded_away_names], axis=1)
        y = np.sign(home_goals - away_goals)

        model = RandomForestRegressor(n_estimators=1000, random_state=42, criterion="friedman_mse")  
        model.fit(x, y)

        return model, team_encoding


    @classmethod
    def train_random_forest_regression_predictor(cls, results: List[Result]) -> Predictor:
        model, team_encoding = cls.build_model(results)

        return cls(model, team_encoding)
