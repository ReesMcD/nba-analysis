import nba.constants as c

import numpy as np
from nba.teams import get_team_opponent_id


def add_game_log_lag(log):
    log = log.sort_values([c.PLAYER_ID, c.GAME_DATE])
    for lag_col in c.LAG_ARR:
        col = lag_col[:-4]
        log[lag_col] = log.groupby(c.PLAYER_ID)[col].shift(1)
    return log


def add_game_log_last_5(log):
    log = log.sort_values([c.PLAYER_ID, c.GAME_DATE])
    for lag_col in c.LAST_5_ARR:
        col = lag_col[:-7]
        log[lag_col] = log[col].rolling(5).mean().shift(1)
    return log


def add_opponent_team_id_to_game_log(log):
    log[c.OPP_TEAM_ID] = np.nan

    for index, row in log.iterrows():
        opp_team_id = get_team_opponent_id(row[c.TEAM_ID], row[c.GAME_ID])
        log.loc[index, c.OPP_TEAM_ID] = opp_team_id

    return log
