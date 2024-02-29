from nba_api.stats.endpoints import boxscoresummaryv2, teamestimatedmetrics, leaguedashlineups
from nba_api.stats.static import teams


def get_teams():
    return teams.get_teams()


def get_team_lineup_stats(team_id, season, date_from, date_to):
    return leaguedashlineups.LeagueDashLineups(
        season=season, team_id_nullable=team_id, date_from_nullable=date_from, date_to_nullable=date_to
    )


def get_team_estimated_metrics(season):
    return teamestimatedmetrics.TeamEstimatedMetrics(season=season)


def get_boxscore_summary(game_id):
    return boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
