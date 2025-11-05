import pandas as pd
import asyncio
import json
from understat_crawling2 import get_game_df
from datetime import datetime
import joblib
import numpy as np
import streamlit as st
#import os
from datetime import datetime, date, time
import pytz
import processing_tools as pt
# os.chdir('C:\\Users\\박진우\\Desktop\\Cap\\data')
# import torch
# from pytorch_tabnet.tab_model import TabNetClassifier

# # 매핑
# team_name_mapping = {
#     'bournemouth':35,
#     'arsenal':42,
#     'aston villa':66,
#     'brentford':  55,
#     'brighton': 51,
#     'burnley': 44,
#     'chelsea': 49,
#     'crystal palace': 52,
#     'everton': 45,
#     'fulham': 36,
#     'leeds': 63,
#     'liverpool': 40,
#     'manchester city': 50,
#     'manchester united': 33,
#     'newcastle united': 34,
#     'nottingham forest':65,    
#     'sunderland':746,
#     'tottenham': 47,
#     'west ham': 48,
#     'wolverhampton wanderers':39,
# }



def epl():

    # Load the logo
    #st.image('EPL/logo.png',width=200)

    st.title('Premier League Football Match Outcome Predictor')

    # Note_message = """The model has been trained on 25 years of historical results (1999-2024). It makes predictions based on past encounters between the teams and their current form. Please note that these predictions are not guaranteed to be accurate and should be used as a guide rather than a definitive forecast. Factors not accounted for by the model can influence match outcomes."""
    
    # st.write("")
    # with st.expander("Note", expanded=False):
    #     st.markdown(Note_message)


    # ======================
    # 날짜와 시간 선택
    # ======================


    with st.container():
        col1, col2 = st.columns(2)  # 한 줄에 두 컬럼으로 배치

        # 날짜 선택
        match_date = col1.date_input("Match date", value=date(2020, 1, 1))


        # 시간 선택
        match_time = col2.time_input("Match time", value=time(12, 0))

    # ======================
    # 시간대 변환 (한국 → 영국)
    # ======================


    x = pd.read_csv('Epl/PL_player.csv',encoding="utf-8")


    # 두 컬럼 생성 (왼쪽: 홈팀, 오른쪽: 원정팀)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Choose home team", list(x['team'].unique()), key="home_team")
        # st.write(f"✅ {home_team} 선수 선택")
        
        selected_home_players =[]
        y=x[x["team"]==home_team][['normalized_player_name','back_number','rating']]


        # 컨테이너 안에 체크박스 넣기
        with st.container():

            # 선수 목록을 리스트로 생성
            player_list = [f"{row['back_number']}. {row['normalized_player_name']}" for _, row in y.iterrows()]

            # multiselect 위젯
            selected_players = st.multiselect(
                "Select players",
                options=player_list,
                default=[],
                key="home_players"
            )

            # 선택 결과를 변수에 저장
            selected_home_players = [(p.split(". ")[1],y[y['normalized_player_name']==p.split(". ")[1]]['rating'].item()) for p in selected_players]  # 번호 제거하고 이름만 저장


    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Choose away team", list(x['team'].unique()), key="away_team")
        # st.write(f"✅ {away_team} 선수 선택")

        selected_away_players =[]
        y=x[x["team"]==away_team][['normalized_player_name','back_number','rating']]

        # 컨테이너 안에 체크박스 넣기
        with st.container():
            # 선수 목록을 리스트로 생성
            player_list = [f"{row['back_number']}. {row['normalized_player_name']}" for _, row in y.iterrows()]

            # multiselect 위젯
            selected_players = st.multiselect(
                "Select players",
                options=player_list,
                default=[],
                key="away_players"
            )

            # 선택 결과를 변수에 저장
            selected_away_players = [(p.split(". ")[1],y[y['normalized_player_name']==p.split(". ")[1]]['rating'].item()) for p in selected_players]  # 번호 제거하고 이름만 저장


    with st.container():
        col1, col2 = st.columns(2)  # 한 줄에 두 컬럼으로 배치

        # 첫 번째 숫자 입력 칸
        num1 = col1.number_input("B365H", step=0.01)

        # 두 번째 숫자 입력 칸
        num2 = col2.number_input("PSH", step=0.01)



    # 결과 출력
    st.markdown("---")
    # st.write("### 최종 선택 결과")
    # st.write(f"**홈 팀 ({home_team}) 선수:** {selected_home_players}")
    # st.write(f"**원정 팀 ({away_team}) 선수:** {selected_away_players}")
    # st.write(f"**경기 날짜 ({match_date}) 경기 시간:** {match_time}")
    # st.write(f"**B365H ({num1}) PSH:** {num2}")


    if st.button('Predict'):
        if ((home_team!=away_team) and 
            selected_home_players and 
            selected_away_players and 
            (match_date!=date(2020, 1, 1)) and
            (match_time!=time(12,0)) and
            num1 and num2):
            with st.spinner('Processing...'):
                processed_date=(pt.t(match_date, match_time)).date() #datetime 반환

                game_df=asyncio.run(get_game_df('epl',2024))
                game_df['home_team'] = game_df['home_team'].str.lower()
                game_df['away_team'] = game_df['away_team'].str.lower()
                game_df['date']=pd.to_datetime(game_df['date'])
                game_df['date']=(game_df['date']).dt.date
                away_prob_5=pt.prob_5(game_df, processed_date, away_team)

                game_df=pd.read_csv("https://www.football-data.co.uk/mmz4281/2425/E0.csv")
                game_df['HomeTeam'] = game_df['HomeTeam'].apply(pt.standardize_team_name)
                game_df['AwayTeam'] = game_df['AwayTeam'].apply(pt.standardize_team_name)
                game_df['Date'] = pd.to_datetime(game_df['Date'], format='%d/%m/%Y')
                game_df['date'] = (game_df['Date']).dt.date
                HTAG_5=pt.htg_5(game_df, processed_date, away_team)

                HSRA=np.mean(([r[1] for r in selected_home_players]))
                ASRA=np.mean(([r[1] for r in selected_away_players]))
                
                df_params={ 'away_prob_5': [away_prob_5], 'HTAG_5':[HTAG_5], 'B365H': [num1],
                          'PSH':[num2],'HSRA':[HSRA], 'ASRA':[ASRA]
                }
                df_params=pd.DataFrame(df_params)

                # 모델 불러와서 변수 넣고 예측하기
                model = joblib.load('Epl/model.joblib')
                scaler = joblib.load('Epl/scaler.joblib')
                df_scaled=scaler.transform(df_params)
                result=model.predict(df_scaled).item()

                # # TabNet 모델
                # model = TabNetClassifier(
                #             n_d=2,
                #             n_a=4,
                #             n_steps=4,
                #             gamma=0.9,
                #             #optimizer_fn=None,
                #             optimizer_params={"lr": 0.04},
                #             mask_type='sparsemax',
                #             # scheduler_params={"step_size":20, "gamma":0.9},
                #             # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                # )

                # # state_dict 로드
                # model.network.load_state_dict(torch.load("Epl/tabnet_model_state.pth", map_location='cpu'))

                # # 모드 전환
                # model.network.eval()
                # result=model.predict(df_params).item()
                # prob=model.predict_proba(df_params)
            st.success('Done!')
            if result == 0:
                st.write(f'The match result prediction: {home_team} wins the match!!')   
                st.balloons()

            elif result == 1:
                st.write(f'The match result prediction: {away_team} wins the match!!')
                st.balloons()
            else:
                st.write(f'The match result prediction: The match ends in a draw!!')    
                
        
        else:
             st.write('Please enter both team names.')        
                




if __name__ == "__main__":
    epl()



# streamlit run "C:\Users\박진우\Desktop\Cap\app.py"








