import streamlit as st
from Epl  import epl
# from LaLiga.laliga import laliga
# from SerieA.seriea import seriea
from home import home_page

def main():
    st.set_page_config(page_title='Football Match Outcome Predictor', page_icon='⚽', layout='centered', initial_sidebar_state='auto')

    st.sidebar.header("Select the League")
    selected_page = st.sidebar.radio("Choose", ["Home", "EPL","La Liga", "Serie A"])

    if selected_page == "Home":
        home_page()
    elif selected_page=="EPL":
        epl()
    # elif selected_page=="La Liga":
    #     laliga()
    # elif selected_page=="Serie A":
    #     seriea()

        
if __name__ == "__main__":
    main()