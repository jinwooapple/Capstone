from datetime import datetime#, date, time
import pytz
import pandas as pd
#import asyncio
import numpy as np

def t(match_date, match_time):
    # 시간대 객체
    kr_tz = pytz.timezone('Asia/Seoul')
    uk_tz = pytz.timezone('Europe/London')

    # 한국 시간 datetime 생성
    kr_dt = datetime.combine(match_date, match_time)
    kr_dt = kr_tz.localize(kr_dt)

    # 영국 시간으로 변환
    uk_dt = kr_dt.astimezone(uk_tz)
    return uk_dt

# understat 데이터로 하는 부분
def prob_5(game_df, match_date, team):
    game_df['home_prob']=pd.to_numeric(game_df['home_prob'])
    game_df['away_prob']=pd.to_numeric(game_df['away_prob'])
    game_df['home_team'] = game_df['home_team'].str.lower()
    game_df['away_team'] = game_df['away_team'].str.lower()
    past_matches = game_df[((game_df['home_team'] == team) | 
                        (game_df['away_team'] == team)) & 
                        (game_df['date'] < match_date)].tail(5)
    
    df1=past_matches[past_matches['home_team']==team]['home_prob'].to_frame().rename(columns={'home_prob': 'val'})
    df2=past_matches[past_matches['away_team']==team]['away_prob'].to_frame().rename(columns={'away_prob': 'val'})
    return pd.concat([df1,df2])['val'].mean()


def xg_5(game_df, match_date, team):
    game_df['xg_home']=pd.to_numeric(game_df['xg_home'])
    game_df['xg_away']=pd.to_numeric(game_df['xg_away'])
    game_df['home_team'] = game_df['home_team'].str.lower()
    game_df['away_team'] = game_df['away_team'].str.lower()
    past_matches = game_df[((game_df['home_team'] == team) | 
                        (game_df['away_team'] == team)) & 
                        (game_df['date'] < match_date)].tail(5)
    
    df1=past_matches[past_matches['home_team']==team]['xg_home'].to_frame().rename(columns={'xg_home': 'val'})
    df2=past_matches[past_matches['away_team']==team]['xg_away'].to_frame().rename(columns={'xg_away': 'val'})
    return pd.concat([df1,df2])['val'].mean()

def pa(game_df, match_date, team):
    past_matches = game_df[((game_df['home_team'] == team) | 
                    (game_df['away_team'] == team)) & 
                    (game_df['date'] < match_date)].tail(5)
    
    past_matches.loc[past_matches['home_team'] == team, 'result'] = (
        past_matches.loc[past_matches['home_team'] == team, 'result']
        .map({'home': 3, 'draw': 1, 'away': 0})
    )
    past_matches.loc[past_matches['away_team'] == team, 'result'] = (
        past_matches.loc[past_matches['away_team'] == team, 'result']
        .map({'home': 0, 'draw': 1, 'away': 3})
    )
    # 데이터 개수에 맞게 가중치 생성 (가장 최근 값이 가장 높은 가중치)
    weights = list(range(1, len(past_matches) + 1))  # [1, 2, ..., N]
    results=past_matches['result'].tolist()
    momentum_bonus=0
    modified_points = []
    for i in range(len(results)):
        if i > 0:
            if results[i] == 3 and results[i-1] == 3:  # 연승
                momentum_bonus += 0.5
            elif results[i] == 0 and results[i-1] == 0:  # 연패
                momentum_bonus -= 0.5
            else:
                momentum_bonus = 0  # 연승/연패가 끊기면 보너스 초기화

        modified_points.append((results[i] + momentum_bonus))

    # 가중평균 계산
    return np.average(modified_points, weights=weights)

def sa(game_df, match_date, team):
    past_matches = game_df[((game_df['home_team'] == team) | 
                    (game_df['away_team'] == team)) & 
                    (game_df['date'] < match_date)].tail(5)
    past_matches.loc[past_matches['home_team'] == team, 'score'] = (
        past_matches.loc[past_matches['home_team'] == team, 'home_score']
    )
    past_matches.loc[past_matches['away_team'] == team, 'score'] = (
        past_matches.loc[past_matches['away_team'] == team, 'away_score']
    )
    # 데이터 개수에 맞게 가중치 생성 (가장 최근 값이 가장 높은 가중치)
    weights = list(range(1, len(past_matches) + 1))  # [1, 2, ..., N]

    # 가중평균 계산
    return np.average(past_matches['score'], weights=weights)





def htg_5(game_df, match_date, team):
    past_matches = game_df[((game_df['HomeTeam'] == team) | 
                    (game_df['AwayTeam'] == team)) & 
                    (game_df['date'] < match_date)].tail(5)
    
    df1=past_matches[past_matches['HomeTeam']==team]['HTHG'].to_frame().rename(columns={'HTHG': 'val'})
    df2=past_matches[past_matches['AwayTeam']==team]['HTAG'].to_frame().rename(columns={'HTAG': 'val'})
    return pd.concat([df1,df2])['val'].median()

def st_5(game_df, match_date, team):
    past_matches = game_df[((game_df['HomeTeam'] == team) | 
                (game_df['AwayTeam'] == team)) & 
                (game_df['date'] < match_date)].tail(5)
    
    df1=past_matches[past_matches['HomeTeam']==team]['HST'].to_frame().rename(columns={'HST': 'val'})
    df2=past_matches[past_matches['AwayTeam']==team]['AST'].to_frame().rename(columns={'AST': 'val'})
    HST_5=pd.concat([df1,df2])['val'].median()



    


# 매핑 적용 함수
def standardize_team_name(team_name):
    team_name_mapping = {
    'Bournemouth': 'bournemouth',
    'Arsenal': 'arsenal',
    'Aston Villa': 'aston villa',
    'Brentford': 'brentford',
    'Brighton': 'brighton',
    'Chelsea': 'chelsea',
    'Crystal Palace': 'crystal palace',
    'Everton': 'everton',
    'Fulham': 'fulham',
    'Leicester': 'leicester',
    'Liverpool': 'liverpool',
    'Man City': 'manchester city',
    'Man United': 'manchester united',
    'Newcastle': 'newcastle united',
    "Nott'm Forest": 'nottingham forest', 
    'Southampton': 'southampton',
    'Tottenham': 'tottenham',
    'West Ham': 'west ham',
    'Wolves': 'wolverhampton wanderers',
    'Ipswich': 'ipswich'
    }


    return team_name_mapping.get(team_name, team_name)  # 매핑된 값이 없으면 기존 값 유지
