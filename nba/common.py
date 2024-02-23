from nba.client.player_client import PlayerClient, get_player_game_log

import nba.constants as c
import pandas as pd


def get_players_stat_std(stat_col, player_list, season_list):
    player_client = PlayerClient()
    columns = [c.ID, c.NAME, stat_col, c.STD, c.VAR]
    seasons = season_list

    df = pd.DataFrame(columns=columns)

    for player_name in player_list:
        player_id = player_client.get_player_id(player_name)

        log = get_player_game_log(player_id, seasons)
        stat_mean = log[stat_col].mean()
        std = log[stat_col].std()
        var = log[stat_col].var()

        temp = pd.DataFrame([[player_id, player_name, stat_mean, std, var]], columns=df.columns)
        df = pd.concat([temp, df], ignore_index=True)

    return df


def add_game_log_lag(log):
    for lag_col in c.LAG_ARR:
        col = lag_col[:-4]
        log[lag_col] = log.groupby(c.PLAYER_ID)[col].shift(1)
    return log


def add_game_log_last_5(log):
    for lag_col in c.LAST_5_ARR:
        col = lag_col[:-7]
        log[lag_col] = log[col].rolling(5).mean().shift(1)
    return log
