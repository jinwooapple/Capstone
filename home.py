import streamlit as st
#from PIL import Image


def home_page():

#    st.image("logo1.png",width=150)

    st.title("Football Match Outcome Predictor ⚽")

    st.write("This web app predicts the outcome of football matches in the top European leagues like: EPL, La Liga, Serie A")

    # st.info("While navigating through different leagues kindly please refresh the site for better performance")
    st.info("The model has been trained on 10 years of historical results (2015-2024). It makes predictions based on past encounters between the teams and their current form.")

    st.write("Select the league from the sidebar to get started")

    container = st.container()

    with container:
        # st.write("* Bundesliga: German Football League")
        # st.text("")
        st.write("* EPL: English Premier League")
        st.text("")
        st.write("* La Liga: Spanish Football League")
        st.text("")
        st.write("* Serie A: Italian Football League")
        st.text("")

    

    # st.success("Thanks for visiting 🤩!!")

if __name__ == "__main__":

    home_page()

