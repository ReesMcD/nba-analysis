# Description: Constants for the NBA API
# Stat constants
ID = 'ID'
PLAYER_ID = 'PLAYER_ID'
PLAYER_NAME = 'PLAYER_NAME'
NAME = 'NAME'

TEAM_ID = 'TEAM_ID'
TEAM_ABBREVIATION = 'TEAM_ABBREVIATION'
TEAM_NAME = 'TEAM_NAME'

SEASON = 'SEASON'
SEASON_ID = 'SEASON_ID'
SEASON_YEAR = 'SEASON_YEAR'

NBA_LEAGUE_ID = '00'

GAME_ID = 'GAME_ID'
GAME_DATE = 'GAME_DATE'
WL = 'WL'
HOME_TEAM_ID = 'HOME_TEAM_ID'
VISITOR_TEAM_ID = 'VISITOR_TEAM_ID'
MATCHUP = 'MATCHUP'

MIN = 'MIN'
FGM = 'FGM'
FGA = 'FGA'
FG_PCT = 'FG_PCT'
FG3M = 'FG3M'
FG3A = 'FG3A'
FG3_PCT = 'FG3_PCT'
FTM = 'FTM'
FTA = 'FTA'
FT_PCT = 'FT_PCT'
OREB = 'OREB'
DREB = 'DREB'
REB = 'REB'
AST = 'AST'
STL = 'STL'
BLK = 'BLK'
TOV = 'TOV'
PF = 'PF'
BLKA = 'BLKA'
PFD = 'PFD'
PTS = 'PTS'
PLUS_MINUS = 'PLUS_MINUS'

MIN_LAG = 'MIN_LAG'
PTS_LAG = 'PTS_LAG'
REB_LAG = 'REB_LAG'
AST_LAG = 'AST_LAG'
STL_LAG = 'STL_LAG'
BLK_LAG = 'BLK_LAG'
TOV_LAG = 'TOV_LAG'
FG_PCT_LAG = 'FG_PCT_LAG'
FGM_LAG = 'FGM_LAG'
FGA_LAG = 'FGA_LAG'
FG3M_LAG = 'FG3M_LAG'
FG3A_LAG = 'FG3A_LAG'
FG3_PCT_LAG = 'FG3_PCT_LAG'
FTM_LAG = 'FTM_LAG'
FTA_LAG = 'FTA_LAG'
FT_PCT_LAG = 'FT_PCT_LAG'
PLUS_MINUS_LAG = 'PLUS_MINUS_LAG'

LAG_ARR = [MIN_LAG, PTS_LAG, REB_LAG, AST_LAG, STL_LAG, BLK_LAG, TOV_LAG, FG_PCT_LAG, FGM_LAG, FGA_LAG, FG3M_LAG,
           FG3A_LAG, FG3_PCT_LAG, FTM_LAG, FTA_LAG, FT_PCT_LAG, PLUS_MINUS_LAG]

MIN_LAST_5 = 'MIN_LAST_5'
PTS_LAST_5 = 'PTS_LAST_5'
REB_LAST_5 = 'REB_LAST_5'
AST_LAST_5 = 'AST_LAST_5'
STL_LAST_5 = 'STL_LAST_5'
BLK_LAST_5 = 'BLK_LAST_5'
TOV_LAST_5 = 'TOV_LAST_5'
FG_PCT_LAST_5 = 'FG_PCT_LAST_5'
FGM_LAST_5 = 'FGM_LAST_5'
FGA_LAST_5 = 'FGA_LAST_5'
FG3M_LAST_5 = 'FG3M_LAST_5'
FG3A_LAST_5 = 'FG3A_LAST_5'
FG3_PCT_LAST_5 = 'FG3_PCT_LAST_5'
FTM_LAST_5 = 'FTM_LAST_5'
FTA_LAST_5 = 'FTA_LAST_5'
FT_PCT_LAST_5 = 'FT_PCT_LAST_5'
PLUS_MINUS_LAST_5 = 'PLUS_MINUS_LAST_5'

LAST_5_ARR = [MIN_LAST_5, PTS_LAST_5, REB_LAST_5, AST_LAST_5, STL_LAST_5, BLK_LAST_5, TOV_LAST_5, FG_PCT_LAST_5,
              FGM_LAST_5, FGA_LAST_5, FG3M_LAST_5, FG3A_LAST_5, FG3_PCT_LAST_5, FTM_LAST_5, FTA_LAST_5, FT_PCT_LAST_5,
              PLUS_MINUS_LAST_5]

GP = 'GP'
W = 'W'
L = 'L'
W_PCT = 'W_PCT'
E_OFF_RATING = 'E_OFF_RATING'
E_DEF_RATING = 'E_DEF_RATING'
E_NET_RATING = 'E_NET_RATING'
E_PACE = 'E_PACE'
E_AST_RATIO = 'E_AST_RATIO'
E_OREB_PCT = 'E_OREB_PCT'
E_DREB_PCT = 'E_DREB_PCT'
E_REB_PCT = 'E_REB_PCT'
E_TM_TOV_PCT = 'E_TM_TOV_PCT'
GP_RANK = 'GP_RANK'
W_RANK = 'W_RANK'
L_RANK = 'L_RANK'
W_PCT_RANK = 'W_PCT_RANK'
MIN_RANK = 'MIN_RANK'
E_OFF_RATING_RANK = 'E_OFF_RATING_RANK'
E_DEF_RATING_RANK = 'E_DEF_RATING_RANK'
E_NET_RATING_RANK = 'E_NET_RATING_RANK'
E_AST_RATIO_RANK = 'E_AST_RATIO_RANK'
E_OREB_PCT_RANK = 'E_OREB_PCT_RANK'
E_DREB_PCT_RANK = 'E_DREB_PCT_RANK'
E_REB_PCT_RANK = 'E_REB_PCT_RANK'
E_TM_TOV_PCT_RANK = 'E_TM_TOV_PCT_RANK'
E_PACE_RANK = 'E_PACE_RANK'

OPP_TEAM_ID = 'OPP_TEAM_ID'
OPP_TEAM_NAME = 'OPP_TEAM_NAME'
OPP_GP = 'OPP_GP'
OPP_W = 'OPP_W'
OPP_L = 'OPP_L'
OPP_W_PCT = 'OPP_W_PCT'
OPP_MIN = 'OPP_MIN'
OPP_E_OFF_RATING = 'OPP_E_OFF_RATING'
OPP_E_DEF_RATING = 'OPP_E_DEF_RATING'
OPP_E_NET_RATING = 'OPP_E_NET_RATING'
OPP_E_PACE = 'OPP_E_PACE'
OPP_E_AST_RATIO = 'OPP_E_AST_RATIO'
OPP_E_OREB_PCT = 'OPP_E_OREB_PCT'
OPP_E_DREB_PCT = 'OPP_E_DREB_PCT'
OPP_E_REB_PCT = 'OPP_E_REB_PCT'
OPP_E_TM_TOV_PCT = 'OPP_E_TM_TOV_PCT'
OPP_GP_RANK = 'OPP_GP_RANK'
OPP_W_RANK = 'OPP_W_RANK'
OPP_L_RANK = 'OPP_L_RANK'
OPP_W_PCT_RANK = 'OPP_W_PCT_RANK'
OPP_MIN_RANK = 'OPP_MIN_RANK'
OPP_E_OFF_RATING_RANK = 'OPP_E_OFF_RATING_RANK'
OPP_E_DEF_RATING_RANK = 'OPP_E_DEF_RATING_RANK'
OPP_E_NET_RATING_RANK = 'OPP_E_NET_RATING_RANK'
OPP_E_AST_RATIO_RANK = 'OPP_E_AST_RATIO_RANK'
OPP_E_OREB_PCT_RANK = 'OPP_E_OREB_PCT_RANK'
OPP_E_DREB_PCT_RANK = 'OPP_E_DREB_PCT_RANK'
OPP_E_REB_PCT_RANK = 'OPP_E_REB_PCT_RANK'
OPP_E_TM_TOV_PCT_RANK = 'OPP_E_TM_TOV_PCT_RANK'
OPP_E_PACE_RANK = 'OPP_E_PACE_RANK'

STD = 'STD'
VAR = 'VARIENCE'

# Parameter constants
PER_GAME = 'PerGame'
PER_36 = 'Per36'
TOTAL = 'Totals'

# Season constants
SEASON_2010_2011 = '2010-11'
SEASON_2011_2012 = '2011-12'
SEASON_2012_2013 = '2012-13'
SEASON_2014_2015 = '2014-15'
SEASON_2015_2016 = '2015-16'
SEASON_2016_2017 = '2016-17'
SEASON_2017_2018 = '2017-18'
SEASON_2018_2019 = '2018-19'
SEASON_2019_2020 = '2019-20'
SEASON_2020_2021 = '2020-21'
SEASON_2021_2022 = '2021-22'
SEASON_2022_2023 = '2022-23'
SEASON_2023_2024 = '2023-24'

# Teams
DET = 'DET'
PHI = 'PHI'

# Player constants
# 76ers
JOEL_EMBIID = 'Joel Embiid'
TYRESE_MAXEY = 'Tyrese Maxey'
# Bucks
GIANIS_ANTETOKOUNMPO = 'Giannis Antetokounmpo'
DAMIAN_LILLARD = 'Damian Lillard'
# Bulls
ZACH_LAVINE = 'Zach Lavine'
DEMAR_DEROZAN = 'DeMar DeRozan'
# Cavaliers
DONOVAN_MITCHELL = 'Donovan Mitchell'
DARIUS_GARLAND = 'Darius Garland'
# Celtics
JAYSON_TATUM = 'Jayson Tatum'
JAYLEN_BROWN = 'Jaylen Brown'
KRISTAAPS_PORZINGIS = 'Kristaaps Porzingis'
# Clippers
PAUL_GEORGE = 'Paul George'
KAWHI_LEONARD = 'Kawhi Leonard'
JAMES_HARDEN = 'James Harden'
# Grizzlies
JA_MORANT = 'Ja Morant'
DESMOND_BANE = 'Desmond Bane'
# Hawks
TRAE_YOUNG = 'Trae Young'
DEJOUNTE_MURRAY = 'Dejounte Murray'
# Heat
BAM_ADEBAYO = 'Bam Adebayo'
JIMMY_BUTLER = 'Jimmy Butler'
TYLER_HERRO = 'Tyler Herro'
# Hornets
TERRY_ROZIER = 'Terry Rozier'
# Rockets
JALEN_GREEN = 'Jalen Green'
# Kings
DEAARON_FOX = 'De\'Aaron Fox'
DOMANTAS_SABONIS = 'Domantas Sabonis'
# Knicks
JALEN_BRUNSON = 'Jalen Brunson'
JULIUS_RANDLE = 'Julius Randle'
# Jazz
LAURI_MARKKANEN = 'Lauri Markkanen'
JORDON_CLARKSON = 'Jordon Clarkson'
# Lakers
LEBRON_JAMES = 'LeBron James'
ANTHONY_DAVIS = 'Anthony Davis'
# Magic
PAOLO_BANCHERO = 'Paolo Banchero'
# Mavericks
LUKA_DONCIC = 'Luka Doncic'
KYRIE_IRVING = 'Kyrie Irving'
# Nuggets
NIKOLA_JOKIC = 'Nikola Jokic'
JAMAL_MURRAY = 'Jamal Murray'
# Nets
MIKAL_BRIDGES = 'Mikal Bridges'
# Pacers
TYRESE_HALIBURTON = 'Tyrese Haliburton'
PASCAL_SIAKAM = 'Pascal Siakam'
# Pelicans
BRAENDON_INGRAM = 'Braendon Ingram'
CJ_MCCOLLUM = 'CJ McCollum'
# Pistons
BOJAN_BOGDANOVIC = 'Bojan Bogdanovic'
# Raptors
# Rockets
FRED_VANVLEET = 'Fred VanVleet'
# Spurs
KELDON_JOHNSON = 'Keldon Johnson'
# Suns
KEVIN_DURANT = 'Kevin Durant'
DEVIN_BOOKER = 'Devin Booker'
# Timberwolves
ANTHONY_EDWARDS = 'Anthony Edwards'
KARL_ANTHONY_TOWNS = 'Karl-Anthony Towns'
# Trail Blazers
JERAMI_GRANT = 'Jerami Grant'
ANFERNEE_SIMONS = 'Anfernee Simons'
# Thunder
SHAI_GILGEOUS_ALEXANDER = 'Shai Gilgeous-Alexander'
# Warriors
STEPHEN_CURRY = 'Stephen Curry'
KLAY_THOMPSON = 'Klay Thompson'
# Wizards
JORDAN_POOLE = 'Jordan Poole'
KYLE_KUZMA = 'Kyle Kuzma'


ALL_STAR_2023_24 = [
    LUKA_DONCIC, KEVIN_DURANT, SHAI_GILGEOUS_ALEXANDER, LEBRON_JAMES, NIKOLA_JOKIC, DEVIN_BOOKER, STEPHEN_CURRY,
    ANTHONY_DAVIS, ANTHONY_EDWARDS, PAUL_GEORGE, KAWHI_LEONARD, KARL_ANTHONY_TOWNS, GIANIS_ANTETOKOUNMPO, JOEL_EMBIID,
    TYRESE_HALIBURTON, DAMIAN_LILLARD, JAYSON_TATUM, BAM_ADEBAYO, PAOLO_BANCHERO, JAYLEN_BROWN, JALEN_BRUNSON,
    TYRESE_MAXEY, DONOVAN_MITCHELL, JULIUS_RANDLE
]

TWENTY_PTS_SCORERS_2022_23 = [
    JOEL_EMBIID, LUKA_DONCIC, DAMIAN_LILLARD, SHAI_GILGEOUS_ALEXANDER, GIANIS_ANTETOKOUNMPO, JAYSON_TATUM,
    DONOVAN_MITCHELL, KYRIE_IRVING, JAYLEN_BROWN, TRAE_YOUNG, JA_MORANT, LAURI_MARKKANEN, JULIUS_RANDLE, DEAARON_FOX,
    ZACH_LAVINE, ANTHONY_EDWARDS, DEMAR_DEROZAN, NIKOLA_JOKIC, PASCAL_SIAKAM, JALEN_BRUNSON, KRISTAAPS_PORZINGIS,
    JIMMY_BUTLER, JALEN_GREEN, KELDON_JOHNSON, KLAY_THOMPSON, DARIUS_GARLAND, BOJAN_BOGDANOVIC, DESMOND_BANE,
    KYLE_KUZMA, TERRY_ROZIER, ANFERNEE_SIMONS, JAMES_HARDEN, CJ_MCCOLLUM, JORDON_CLARKSON, JERAMI_GRANT,
    DEJOUNTE_MURRAY, JORDAN_POOLE, BAM_ADEBAYO, TYRESE_MAXEY, MIKAL_BRIDGES, TYLER_HERRO, JAMAL_MURRAY, PAOLO_BANCHERO
]

# GAME_LOG mask
PLAYER_GAME_LOG_COLUMNS = [
    SEASON_YEAR, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP, WL, MIN,
    FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, BLKA, PF, PFD, PTS,
    PLUS_MINUS
]

GAME_LOG_LAG_MASK = [
    MIN_LAG,
    PTS_LAG,
    REB_LAG,
    AST_LAG,
    STL_LAG,
    BLK_LAG,
    TOV_LAG,
    FG_PCT_LAG,
    FGM_LAG,
    FGA_LAG,
    FG3M_LAG,
    FG3A_LAG,
    FG3_PCT_LAG,
    FTM_LAG,
    FTA_LAG,
    FT_PCT_LAG,
    PLUS_MINUS_LAG,
]

GAME_LOG_LAST_5_MASK = [
    MIN_LAST_5,
    PTS_LAST_5,
    REB_LAST_5,
    AST_LAST_5,
    STL_LAST_5,
    BLK_LAST_5,
    TOV_LAST_5,
    FG_PCT_LAST_5,
    FGM_LAST_5,
    FGA_LAST_5,
    FG3M_LAST_5,
    FG3A_LAST_5,
    FG3_PCT_LAST_5,
    FTM_LAST_5,
    FTA_LAST_5,
    FT_PCT_LAST_5,
    PLUS_MINUS_LAST_5
]

PTS_PREDICTION_MASK = [PTS] + GAME_LOG_LAG_MASK + GAME_LOG_LAST_5_MASK
