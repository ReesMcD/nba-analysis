import nba.client.team_client as tc
import nba.constants as c


def get_team_id(abbreviation):
    teams = tc.get_teams()
    return [team for team in teams if team[c.ABBREVIATION_PARAM] == abbreviation][0][c.ID_PARAM]


def get_team_lineup_stats(team_id, season, date_from, date_to):
    return tc.get_team_lineup_stats(team_id, season, date_from, date_to).get_data_frames()[0]


def get_team_estimated_metrics(season):
    return tc.get_team_estimated_metrics(season).get_data_frames()[0]


def get_team_opponent_id(team_id, game_id):
    df = tc.get_boxscore_summary(game_id).get_data_frames()[0]
    id_arr = df[[c.HOME_TEAM_ID, c.VISITOR_TEAM_ID]].values.tolist()[0]
    id_arr.remove(team_id)
    return id_arr[0]
