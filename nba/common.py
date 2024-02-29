import nba.constants as c


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
