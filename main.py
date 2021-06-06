#%%
import copy
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import  load_compute_data, filter_sort_selection, make_table_header, make_table, df_nom_dep

clages_selected = {
    '20 ans et plus': 0,
    '20 - 29 ans': 29, 
    '30 - 39 ans': 39, 
    '40 - 49 ans': 49, 
    '50 - 59 ans': 59,
    '60 - 69 ans': 69,
    '70 - 79 ans': 79,
    '80 ans et plus' : 80
}

# sélection des départements inclus tous les départements + Tous les départemnts (renvoie 'every)
dep_selected = pd.concat([pd.Series(index=['Tous les départements'], data='every', dtype=str), pd.Series(index=df_nom_dep.values, data=df_nom_dep.index)])
#%%
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
def load():
    return load_compute_data()

def filter(df, dep='every', age=None):
    return filter_sort_selection(df, dep=dep, age=age)

# def display(df, dep, age):
#     fig_header = make_table_header(dep=dep, age=age)
#     fig_table  = make_table(df, dep=dep, age=age)
#     st.pyplot(fig_header)
#     # st.pyplot(fig)


result = load()

with st.form('my form'):
    clage = st.selectbox("Classe d'âge", list(clages_selected.keys()))
    age = clages_selected[clage]
    dep = st.selectbox("Département", list(dep_selected.index))
    dep = dep_selected[dep]
    submitted = st.form_submit_button('Submit')


if submitted:
    data = filter(result, dep, age)
    st.pyplot(make_table_header(dep, age))
    st.pyplot(make_table(data, dep, age))


# fig_header, ax = plt.subplots()
# st.pyplot(make_table_header(dep, age))
# st.pyplot(fig_table)

# fig_header = make_table_header(dep, age)
# st.pyplot(make_tablefilter_sort_selection(result, dep, age))

#%%
# dep_selected

# with st.spinner("Le chargement des données est en cours. \n Encore un peu de patience avant de pouvoir les explorer!"):
#     result = copy.deepcopy(computations())
# with st.spinner("Calcul du tableau pour les départements. Cela peut prendre du temps..."):
#     fig_header, fig_table = make_table(result, clages_selected[age])
# st.pyplot(fig_header)
# st.pyplot(fig_table)
# 
# %%
