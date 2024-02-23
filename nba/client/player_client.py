from nba_api.stats.endpoints import playercareerstats, playerprofilev2, playergamelogs, leaguedashplayerstats
from nba_api.stats.static import players

from nba.client.team_client import get_team_opponent_id

import nba.constants as c
import pandas as pd
import numpy as np

# TODO: Refactor to service and client pattern


def get_player_team_id(player_id, season):
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    career_df = career.get_data_frames()[0]
    season_df = career_df[career_df[c.SEASON_ID] == season]
    team_df = season_df[season_df[c.TEAM_ID] != 0][c.TEAM_ID].reset_index(drop=True)

    return team_df.values.tolist()


def get_player_profile(player_id, per_mode=c.PER_GAME):
    return playerprofilev2.PlayerProfileV2(player_id=player_id, per_mode36=per_mode).get_data_frames()[0]


def get_player_game_log(player_id, season):
    logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]

    player_name = logs[c.PLAYER_NAME].values[0]
    logs = logs[c.PLAYER_GAME_LOG_COLUMNS]

    logs[c.GAME_DATE] = pd.to_datetime(logs[c.GAME_DATE])
    logs[c.SEASON] = season
    logs[c.OPP_TEAM_ID] = np.nan

    for index, row in logs.iterrows():
        opp_team_id = get_team_opponent_id(row[c.TEAM_ID], row[c.GAME_ID])
        logs.loc[index, c.OPP_TEAM_ID] = opp_team_id

    logs.to_csv('./data/game_logs/{season}/player/{name}_{season}_game_log.csv'.format(name=player_name, season=season), index=False)
    return logs


def get_player_career_stats(player_id):
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    return career.get_data_frames()[0]


def get_all_players_by_season(season):
    return leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]


class PlayerClient:
    full_name = 'full_name'
    id = 'id'

    def __init__(self):
        self.players = players.get_players()
        print('Number of players fetched: {}'.format(len(self.players)))

    def get_player_id(self, name):
        return [player for player in self.players if player[self.full_name] == name][0][self.id]

    def get_player_name(self, player_id):
        return [player for player in self.players if player[self.id] == player_id][0][self.full_name]

    def get_players_game_log(self, player_arr, season, title):
        game_logs = pd.DataFrame()

        for player in player_arr:
            player_id = self.get_player_id(player)
            df = get_player_game_log(player_id, season)
            game_logs = pd.concat([game_logs, df], ignore_index=True)
            print('{player} game logs fetched'.format(player=player))

        game_logs.to_csv('./data/game_logs/{season}/{title}_{season}_game_log'.format(title=title, season=season), index=False)
        return game_logs







