#%%
import copy
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import compute_all, load_format_data, make_table

clages_selected = {
    "Tout âge à partir de 24 ans": None, 
    '24 - 29 ans': 29, 
    '30 - 39 ans': 39, 
    '40 - 49 ans': 49, 
    '50 - 59 ans': 59,
    '60 - 64 ans': 64,
    '65 - 69 ans': 69,
    '70 - 74 ans': 74,
    '75 - 79 ans': 79,
    '80 ans et plus' : 80
}
@st.cache
def computations():
    result = load_format_data()
    return result

def display_table(result, age=None):
    make_table(result, age)

with st.spinner("Le chargement des données est en cours. \n Encore un peu de patience avant de pouvoir les explorer!"):
    result = copy.deepcopy(computations())
age = st.selectbox("Classe d'âge", list(clages_selected.keys()))
with st.spinner("Calcul du tableau pour les 101 départements"):
    fig_header, fig_table = make_table(result, clages_selected[age])
st.pyplot(fig_header)
st.pyplot(fig_table)
# 