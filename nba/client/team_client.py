from nba_api.stats.endpoints import boxscoresummaryv2, teamestimatedmetrics, leaguedashlineups
from nba_api.stats.static import teams

import nba.constants as c

# TODO: Refactor to service and client pattern


def get_team_lineup_stats(team_id, season, date_from, date_to):
    return leaguedashlineups.LeagueDashLineups(
        season=season, team_id_nullable=team_id, date_from_nullable=date_from, date_to_nullable=date_to
    ).get_data_frames()[0]


def get_team_estimated_metrics(season):
    return teamestimatedmetrics.TeamEstimatedMetrics(season=season).get_data_frames()[0]


def get_team_opponent_id(team_id, game_id):
    df = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id).get_data_frames()[0]
    id_arr = df[[c.HOME_TEAM_ID, c.VISITOR_TEAM_ID]].values.tolist()[0]
    id_arr.remove(team_id)
    return id_arr[0]


class TeamClient:
    id = 'id'
    abbreviation = 'abbreviation'

    def __init__(self):
        self.teams = teams.get_teams()
        print('Number of teams fetched: {}'.format(len(self.teams)))

    def get_teams(self):
        return self.teams

    def get_team_id(self, abbreviation):
        return [team for team in self.teams if team[self.abbreviation] == abbreviation][0][self.id]
