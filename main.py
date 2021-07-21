#%%
import copy
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import  load_data, compute_data, filter_sort_selection, make_table_header, make_table, df_nom_dep
from texte import display_changelog, display_desc, display_att

clages_selected = {
    '18 ans et plus': 0,
    '18 - 29 ans': 29, 
    '30 - 39 ans': 39, 
    '40 - 49 ans': 49, 
    '50 - 59 ans': 59,
    '60 - 69 ans': 69,
    '70 - 79 ans': 79,
    '80 ans et plus' : 80
}

indicateurs = {
    '% de couverture complète' : 'couv_complet', 
    '% de primo-vaccinations': 'nouveaux', 
    "% d'injections effectuées" : 'ratio', 
    "Incidence parmi les non vaccinés" : 'inc_S'
}

url_vacc = 'https://www.data.gouv.fr/fr/datasets/r/83cbbdb9-23cb-455e-8231-69fc25d58111'

# sélection des départements inclus tous les départements + Tous les départemnts (renvoie 'every)
dep_selected = pd.concat([pd.Series(index=['Tous les départements'], data='every', dtype=str), pd.Series(index=df_nom_dep.values, data=df_nom_dep.index)])
#%%
st.set_page_config(page_title="Tableau de bord vaccination", page_icon="📈")

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

displayed = False
table_title = "France - 18 ans et plus"

top_container = st.beta_container()
table_container = st.beta_container()
bottom_container = st.beta_container()

data = copy.deepcopy(load_data(url_vacc))
result = compute_data(data[0], data[1], data[2])


def display(df, dep, age):
    global displayed, table_title, result
    displayed = True
    data = filter_sort_selection(result, dep, age, sorting=indicateurs[sort_by])
    st.sidebar.write(f"Dernières données disponibles : {data['last_date']:%d-%b-%Y}")
    if dep == 'every':
        table_title = f'Tous les départements - {clage_str}'
    else:
        table_title = f"{dep_str} - 18 ans et plus"
    st.subheader(table_title)
    st.pyplot(make_table_header(dep, age))
    with st.spinner("Le tableau est en train de charger, encore un peu de patience...⏱️"):
        st.pyplot(make_table(data, dep, age))

# sidebar
with st.sidebar.form('my form'):
    st.write("Sélectionner une classe d'âge ou un département")
    clage_str = st.selectbox("Classe d'âge", list(clages_selected.keys()))
    age = clages_selected[clage_str]
    dep_str = st.selectbox("Département", list(dep_selected.index), index=1)
    dep = dep_selected[dep_str]
    sort_by = st.selectbox("Classer les départements par", list(indicateurs.keys()))
    submitted = st.form_submit_button('Afficher')
st.sidebar.markdown("*Pour «Tous les départements», le délai d'affichage est un peu plus long, pas d'inquiétude si le tableau n'apparaît pas tout de suite.*")

# above table
with top_container:
    st.title("Tableau de bord de suivi de la vaccination contre le Covid-19 en France")
    display_att()

with table_container:
    if submitted:
        filtered = filter_sort_selection(result, dep=dep, age=age)
        display(filtered,dep, age)
        first_load = False

with bottom_container:
    st.markdown(
        '''
        Sources : Santé Publique France - [Vaccination](https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/) - [Incidence et population](https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/)

        '''
    )
    display_desc()
    display_changelog()

if displayed:
    st.stop()
with table_container:
    display(result, dep, age)
