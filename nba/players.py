import nba.client.player_client as pc
import nba.constants as c
import pandas as pd
import numpy as np

from nba.teams import get_team_opponent_id


def get_player_profile(player_id, per_mode=c.PER_GAME):
    return pc.get_player_profile(player_id, per_mode).get_data_frames()[0]


def get_player_career_stats(player_id):
    return pc.get_player_career_stats(player_id).get_data_frames()[0]


def get_all_players_by_season(season, per_mode=c.PER_GAME):
    return pc.get_all_players_by_season(season, per_mode).get_data_frames()[0]


def get_player_team_id(player_id, season):
    career = pc.get_player_career_stats(player_id).get_data_frames()[0]
    season_df = career[career[c.SEASON_ID] == season]
    team_df = season_df[season_df[c.TEAM_ID] != 0][c.TEAM_ID].reset_index(drop=True)

    return team_df.values.tolist()


def get_player_id(name):
    players = pc.get_all_players()
    return [player for player in players if player[c.FULL_NAME_PARAM] == name][0][c.ID_PARAM]


def get_player_name(player_id):
    players = pc.get_all_players()
    return [player for player in players if player[c.ID_PARAM] == player_id][0][c.FULL_NAME_PARAM]


def get_player_game_log(player_id, season):
    logs = pc.get_player_game_log(player_id, season).get_data_frames()[0]

    logs = logs[c.PLAYER_GAME_LOG_COLUMNS]

    logs[c.GAME_DATE] = pd.to_datetime(logs[c.GAME_DATE])
    logs[c.SEASON] = season
    # logs[c.OPP_TEAM_ID] = np.nan

    # TODO: Remove this and add it as its own function
    # for index, row in logs.iterrows():
    #     opp_team_id = get_team_opponent_id(row[c.TEAM_ID], row[c.GAME_ID])
    #     logs.loc[index, c.OPP_TEAM_ID] = opp_team_id

    return logs


def get_players_game_log(player_id_arr, season):
    game_logs = pd.DataFrame()
    count = 1

    for player_id in player_id_arr:
        df = get_player_game_log(player_id, season)
        game_logs = pd.concat([game_logs, df], ignore_index=True)
        print('{count}: {player_id} game logs fetched'.format(count=count, player_id=player_id))
        count += 1

    return game_logs
