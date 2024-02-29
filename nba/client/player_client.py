from nba_api.stats.endpoints import playercareerstats, playerprofilev2, playergamelogs, leaguedashplayerstats
from nba_api.stats.static import players


def get_player_game_log(player_id, season):
    return playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season)


def get_player_profile(player_id, per_mode):
    return playerprofilev2.PlayerProfileV2(player_id=player_id, per_mode36=per_mode)


def get_player_career_stats(player_id):
    return playercareerstats.PlayerCareerStats(player_id=player_id)


def get_all_players_by_season(season, per_mode):
    return leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed=per_mode)


def get_all_players():
    return players.get_players()
