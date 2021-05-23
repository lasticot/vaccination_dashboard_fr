#%%
from collections import OrderedDict
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#######
# chargement et formattage des data
#######

def load_format_data():
    # vaccination
    df1 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/83cbbdb9-23cb-455e-8231-69fc25d58111', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str}, usecols=['dep', 'jour', 'clage_vacsi', 'n_dose1', 'n_complet'])

    # population par département, classe d'âge et sexe
    # source INSEE : https://www.insee.fr/fr/statistiques/1893198
    df2 = pd.read_excel('pop_dep_sexe_age.xlsx', engine='openpyxl', dtype={'dep':str})

    # changement de nom
    vacc = df1.copy()
    departements = df2.copy()

    # Unpivot du fichier dep
    departements = departements.melt(id_vars = ['dep', 'nom_dep'], value_name='pop')

    # split du sexe et de la classe d'âge
    departements['sexe']  = departements.variable.str[0]
    departements['age'] = departements.variable.str[1:]
    departements.drop('variable', axis=1, inplace=True)

    # on laisse tomber les <25 pour faire correspondre les classes d'âges dans les 2 df
    departements.age = departements.age.str.strip('+').astype(int)
    departements = departements[departements.age >= 29].copy()
    vacc = vacc[vacc.clage_vacsi >= 29].copy()

    # on laisse tomber le dep 00 (total) et >976 (pas présents dans dep)
    vacc = vacc[vacc.dep.isin(departements.dep.unique())].copy()

    # regroupement des classes d'âges dans dep pour correspondre aux classes d'âges des vaccins
    bins = pd.IntervalIndex.from_tuples([(24, 29), (30, 39), (40, 49), (50, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, np.inf)])
    departements['clage'] = pd.cut(departements.age, bins, labels=['29', '39', '49', '59', '64', '69', '74', '79', '89'])
    temp = departements.groupby(['dep', 'clage'], as_index=False)['pop'].sum()

    renaming = dict(zip(temp.clage.unique(), vacc.clage_vacsi.unique()))
    temp['clage'] = temp['clage'].replace(renaming)

    # ajout des noms de département (le gropuby ne marche pas si je les garde ¯\_(ツ)_/¯)
    nom_dep = departements.groupby(['dep', 'nom_dep'], as_index=False).min()[['dep', 'nom_dep']]
    departements = temp.merge(nom_dep, how='left', on='dep')

    return vacc, departements

###########
# Values computation
###########
def compute_vacc(vacc, departements, dep, age=None):
    '''
    Returns dictionary of vaccination values needed to draw bullet graphs and sparkline
    aggregated by department and age. 
    keys : 'population', 'dep_nom', 'pc_dose1', 'pc_complet', 'mm_injections', 'inj_last_7D', 'inj_minus_1W'
    ++++++++++++++ Ajouter le calcul de l'objectif de 35M en juin +++++++++++
    '''

    # dep population and name
    if age == None:
        dep_age = departements[['dep', 'pop', 'nom_dep']].groupby(['dep', 'nom_dep'], as_index=False).sum()
        dep_age = dep_age[dep_age['dep'] == dep].copy()
        v_dep_age = vacc[['dep', 'jour', 'n_dose1', 'n_complet']].groupby(['dep', 'jour'], as_index=False).sum()
        v_dep_age = v_dep_age[v_dep_age['dep'] == dep].copy()
    else:
        dep_age = departements[(departements['clage'] == age) & (departements['dep'] == dep)].copy()
        v_dep_age = vacc[(vacc.clage_vacsi == age) & (vacc.dep == dep)][['jour', 'n_dose1', 'n_complet']].copy()

    dep_pop = dep_age['pop'].iloc[0]
    dep_nom = str(dep + ' - ' + dep_age['nom_dep'].iloc[0])

    # filter dep and age in vacc

    # Total number of injections by day 
    v_dep_age['tot_inj'] = v_dep_age['n_dose1'] + v_dep_age['n_complet']
    injections = v_dep_age[['jour', 'tot_inj']].copy().set_index('jour')
    injections['mm7D'] = injections.rolling(7).mean()
    # Keep last 30 days
    injections = injections.tail(30).copy()

    # nb of injections avg last 7 days
    inj_last_7D = injections['mm7D'].iloc[-1]
    # nb of injections avg 7 days before last 7 days
    inj_last_7D_minus1W = injections['mm7D'].iloc[-8]

    # total vaccinated
    total_dose1 = v_dep_age['n_dose1'].sum()
    total_complet =  v_dep_age['n_complet'].sum()

    # vaccinated %
    pc_dose1 = total_dose1 / dep_pop
    pc_complet = total_complet / dep_pop

    return {
        'population'    : dep_pop,
        'dep_nom'       : dep_nom,
        'pc_dose1'      : pc_dose1,
        'pc_complet'    : pc_complet,
        'mm_injections' : injections['mm7D'],
        'inj_last_7D'   : inj_last_7D,
        'inj_minus_1W'  : inj_last_7D_minus1W
    }
    
def compute_vacc_fr(vacc, departements, age=None): 
    '''
    returns dictionary of vacc values for France
    keys : 
    'population', 'pc_dose1', 'pc_complet', 'mm_injections', 'inj_minus_1W'
    '''

    # total population for age group
    if age == None:
        fr_pop = departements['pop'].sum()
        v_fr_age = vacc.groupby('jour', as_index=False).sum()[['jour', 'n_dose1', 'n_complet']]
        v_fr_age['tot_inj'] = v_fr_age['n_dose1'] + v_fr_age['n_complet']
        injections = v_fr_age[['jour', 'tot_inj']].copy().set_index('jour')
    else:
        fr_age = departements.groupby('clage').sum()
        fr_pop = fr_age.loc[age,'pop']
        # group by age in vacc
        v_fr = vacc.groupby(['jour', 'clage_vacsi'], as_index=False).sum()[['jour', 'clage_vacsi', 'n_dose1', 'n_complet']]
        # filtrer la classe d'âge sélectionnoée
        v_fr_age = v_fr[v_fr['clage_vacsi'] == age].copy()
        # Total number of injections
        v_fr_age['tot_inj'] = v_fr_age['n_dose1'] + v_fr_age['n_complet']
        injections = v_fr_age[['jour', 'tot_inj']].copy().set_index('jour')

    injections['mm7D'] = injections.rolling(7).mean()
    # keep last 30 days
    injections = injections.tail(30).copy()
     
    # nb of injections avg last 7 days
    inj_last_7D = injections['mm7D'].iloc[-1]
    # nb of injections avg 7 days before last 7 days
    inj_last_7D_minus1W = injections['mm7D'].iloc[-8]

    # total vaccinated
    total_dose1 = v_fr_age['n_dose1'].sum()
    total_complet =  v_fr_age['n_complet'].sum()

    # vaccinated %
    if fr_pop != 0:
        pc_dose1 = total_dose1 / fr_pop
        pc_complet = total_complet / fr_pop
    else:
        pc_dose1, pc_complet = 'nan', 'nan'

    return {
        'population'    : fr_pop,
        'pc_dose1'      : pc_dose1,
        'pc_complet'    : pc_complet,
        'mm_injections' : injections['mm7D'],
        'inj_last_7D'   : inj_last_7D,
        'inj_minus_1W'  : inj_last_7D_minus1W
    }

def sort_dep(departements, vacc, age=None):
    '''
    Sort dep by pc of dose1
    '''

    all_dep_complet = []

    for dep in vacc.dep.unique():
        all_dep_complet.append(compute_vacc(vacc, departements, dep, age)['pc_complet'])
    dep_names = vacc.dep.unique()
    dep_names_doses = OrderedDict(zip(dep_names, all_dep_complet))
    dep_names_doses_sorted = {k:v for k,v in sorted(dep_names_doses.items(), key=lambda item : item[1], reverse=True)}

    return dep_names_doses_sorted

################
# bullet chart
################

def make_bullet(dict_fr, dict_dep = None, dose=1, first_col=False):
    '''
    Returns a bullet chart go 
    '''     
    if dose==1:
        if dict_dep == None:
            score, target = dict_fr['pc_dose1'], 1
        else:
            score, target = dict_dep['pc_dose1'], dict_fr['pc_dose1']
    elif dose==2:
        if dict_dep == None:
            score, target = dict_fr['pc_complet'], 1
        else:
            score, target = dict_dep['pc_complet'], dict_fr['pc_complet']
    
    if dict_dep == None:
        dep_nom = 'France'
    else:
        dep_nom = dict_dep['dep_nom']
        dep_nom = textwrap.wrap(dep_nom, width=15)
        dep_nom = '<br>'.join(dep_nom)

    # global quartile_1, quartile_complet
    chart = go.Indicator(
        value = score,
        number = {'font' : {'size' : 14}, 'valueformat' : '0%'},
        mode = 'gauge+number', 
        # title = 'nom du departement', 
        gauge = {
            'shape' : 'bullet',
            'borderwidth' : 0.2,
            'bordercolor' : 'blue',
            'bar' : {'color' : '#4259D6', 'thickness' : 0.6, 'line' : {'width' : 0}},
            'axis' : {'visible': False, 'range' : [0, 1]},
            'threshold' : {
                'line' : {'width' : 2}, 
                'value' : target,
                'thickness' : 0.8
            },
            'steps' : [
                {'range' : [0, 1], 'color' : '#ECEFFE'},
            #     {'range' : [quartile_1[0], quartile_1[2]], 'color' : '#D8DEFE'},
            #     {'range' : [quartile_1[2], 1], 'color' : '#ECEFFE'}
            ]
        },
        domain = {'x' : [0,1], 'y' : [0,1]}
    )
    if dose == 1:
        chart['title'] = {'text' : dep_nom, 'align' : 'left', 'font' : {'size' : 14}}
    return chart

##############
#Sparklines
#############

def make_sparkline(dict):
    '''
    Returns a sparkline go
    '''
    df = dict['mm_injections']
    # global values
    sparkline = go.Scatter(
        x = df.index,
        y = df,
        marker_color = '#4259D6',
        fill = 'tozeroy',
        fillcolor = 'rgba(66, 89, 214, 0.2)',
        showlegend = False
    )

    last_point = go.Scatter(
        x = [df.index[-1]],
        y = [df.iloc[-1]],
        mode = 'markers',
        marker = {'size' : 5, 'color' : '#4259D6' },
        showlegend = False
    )
    return sparkline, last_point

################
# Card
################

def make_card(dict):
    last_week = dict['inj_last_7D']
    minus_1W = dict['inj_minus_1W']

    card = go.Indicator(
        value = last_week,
        number = {'font' : {'size' : 14}}, #'valueformat' : '>,.0f'},
        delta = {'reference' : minus_1W, 'position' : 'bottom',
                'relative' : True, 'valueformat' : '.0%', 'font' : {'size' : 10}},
        mode = 'number+delta',
        align = 'right'

    )
    return card

def make_header(title):
    header = go.Indicator(
        title = 'ceci est un titre', 
        mode = 'number'
    )

###############
# Full table
###############

def make_table(age=None):

    vacc, departements = load_format_data()

    n_rows = 10 * 3 + 3 # departements.shape[0] * 3 + 1
    n_cols = 4

    row_specs = [[None               , None               , {'type' : 'xy', 'rowspan' : 3}, {'type' : 'domain', 'rowspan' : 3}],
                 [{'type' : 'domain'}, {'type' : 'domain'}, None                          , None                              ],
                 [None               , None               , None                          , None                              ]],
    
    specs = row_specs[0] * 11
    row_heights = [1, 4.5, 1] * 11

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs = specs,
        column_widths = [2.5, 2.5, 1, 0.5],
        row_heights = row_heights,
        horizontal_spacing = 0.03,
        vertical_spacing   = 0.02,
        print_grid = False
    )
    dict_fr = compute_vacc_fr(vacc, departements, age)

    def make_row(vacc, departements, dep_idx, fig=fig, dep=None, age=age):

        if dep != None:
            dict_dep = compute_vacc(vacc, departements, dep, age)
            dep_nom = dict_dep['dep_nom']
            # middle row in the department's row (1st row is for Fr, each dep's row takes 3 rows in the grid)
            mid_row = (dep_idx  + 1) * 3 + 2

            fig.append_trace(
                make_bullet(dict_fr, dict_dep),
                mid_row, 1)
            dep_nom = textwrap.wrap(dep_nom, width=15)
            dep_nom = '<br>'.join(dep_nom)
            # fig.update_traces(
            #     title = {'text' : dep_nom, 'align' : 'left', 'font' : {'size' : 14}}, 
            # )
            fig.append_trace(
                make_bullet(dict_fr, dict_dep, dose=2),
                mid_row, 2
            )
            fig.append_trace(
                make_sparkline(dict_dep)[0],
                mid_row-1, 3
            )
            fig.update_xaxes(visible=False, showgrid=False)
            fig.update_yaxes(visible=False, showgrid=False)
            fig.add_trace(make_sparkline(dict_dep)[1], mid_row-1, 3)
            min_inj = min(dict_dep['mm_injections']) - 100
            max_inj = max(dict_dep['mm_injections']) + 100
            fig.update_yaxes(range=[min_inj, max_inj], row=mid_row-1, col=3)
            fig.append_trace(make_card(dict_dep), mid_row-1,4)
        else:
            dep_nom = "France"
            mid_row = 2
            fig.append_trace(
                make_bullet(dict_fr),
                mid_row, 1
            )
            fig.append_trace(
                make_bullet(dict_fr, dose=2),
                mid_row, 2 
            )
            fig.append_trace(
                make_sparkline(dict_fr)[0],
                mid_row-1, 3
            )
            fig.add_trace(make_sparkline(dict_fr)[1])
            min_inj = min(dict_fr['mm_injections']) - 500
            max_inj = max(dict_fr['mm_injections']) + 500
            fig.update_yaxes(range=[min_inj, max_inj], row=mid_row-1, col=3)
            fig.append_trace(make_card(dict_fr), mid_row-1,4)
    
    sorted_dep = sort_dep(departements, vacc, age)
    listing = list(sorted_dep.keys())[:10]

    # France total row
    make_row(vacc, departements, -1)

    # departements rows
    for idx, dep in enumerate(listing):
        make_row(vacc, departements, idx, dep=dep)

    fig.update_layout(
        height= 500,
        width = 750,
        margin = {
            'l' : 120, 
            'r' : 30, 
            't' : 20,
            'b' : 5},
        xaxis = {'visible' : False, 'showgrid':True, 'gridwidth':0},
        yaxis = {'visible' : False, 'showgrid':True},
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)'
        )
    
    bckgr=[
        dict(
            type = 'rect',
            xref = 'paper',
            yref = 'paper', 
            x0 = -0.5,
            x1 = 1.2,
            y0 = i/11,
            y1 = (i+1)/11,
            fillcolor = 'grey',
            layer = 'below',
            line_width = 0,
            opacity = 0.1
        )
    for i in range(1,11,2)
    ]
    bckgr.append(
            dict(
                type = 'rect',
                xref = 'paper',
                yref = 'paper',
                x0 = -1.5,
                x1 = 1.5,
                y0 = 1.015,
                y1 = 1.2,
                fillcolor = 'darkblue', 
                layer = 'below',
                line_width = 0,
                opacity = 0.5
            )
    )
    fig.update_layout(
        shapes=bckgr
    )

    fig.update_layout(
        margin = dict(
            t = 80
        )
    )


    fig.add_annotation(
        text = "Pourcentage de la population<br>ayant reçu...",
        showarrow=False,
        font = dict(
            color = 'white',
            size=14
        ),
        xref = 'paper', 
        yref = 'paper',
        x = 0.35,
        y = 1.20
    )
    fig.add_annotation(
        text = '... 1 dose',
        showarrow=False,
        font = dict(
            color = 'white',
            size=14
        ),
        xref = 'paper',
        yref = 'paper', 
        x = 0.05,
        y = 1.1
    )
    fig.add_annotation(
        text = '... couverture complète',
        showarrow=False,
        font = dict(
            color = 'white',
            size=14
        ),
        xref = 'paper',
        yref = 'paper', 
        x = 0.5,
        y = 1.1
    )
    fig.add_annotation(
        text = "Nb d'injections<br>(Moy. mobile 7 jrs)",
        showarrow=False,
        font = dict(
            color = 'white',
            size=13
        ),
        xref = 'paper',
        yref = 'paper', 
        x = 1.02,
        y = 1.2
    )
    fig.add_annotation(
        text = "30 derniers<br>jours",
        showarrow=False,
        font = dict(
            color = 'white',
            size=11
        ),
        xref = 'paper',
        yref = 'paper', 
        x = 0.87,
        y = 1.1
    )
    fig.add_annotation(
        text = "7 dern. jrs<br>% p.r 7 jrs préc.",
        showarrow=False,
        font = dict(
            color = 'white',
            size=11
        ),
        xref = 'paper',
        yref = 'paper', 
        x = 1.05,
        y = 1.1
    )
    fig.show()
