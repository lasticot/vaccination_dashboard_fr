#%%
import copy
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import  load_format_data, make_table

clages_selected = {
    '20 - 29 ans': 29, 
    '30 - 39 ans': 39, 
    '40 - 49 ans': 49, 
    '50 - 59 ans': 59,
    '60 - 69 ans': 69,
    '70 - 79 ans': 79,
    '80 ans et plus' : 80
}
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1000px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
@st.cache
def computations():
    result = load_format_data()
    return result

def display_table(result, age=None):
    make_table(result, age)

with st.spinner("Le chargement des données est en cours. \n Encore un peu de patience avant de pouvoir les explorer!"):
    result = copy.deepcopy(computations())
age = st.selectbox("Classe d'âge", list(clages_selected.keys()))
with st.spinner("Calcul du tableau pour les départements. Cela peut prendre du temps..."):
    fig_header, fig_table = make_table(result, clages_selected[age])
st.pyplot(fig_header)
st.pyplot(fig_table)
# 