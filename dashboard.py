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
def compute_vacc(vacc, dep_row):
    '''
    Iterated over rows of 'departements
    Returns dictionary of vaccination values for each dep and age 
    keys : 'population', 'dep', 'nom_dep', 'age', 'total_dose1', 'total_complet', 'injections'(Series)
    '''
    dep = dep_row['dep']
    age = dep_row['clage']
    # filter vacc on  dep and age
    v_dep_age = vacc[(vacc['dep'] == dep) & (vacc['clage_vacsi'] == age)].copy()

    # total vaccinated
    total_dose1 = v_dep_age['n_dose1'].sum()
    total_complet =  v_dep_age['n_complet'].sum()

    # Total number of injections by day 
    v_dep_age['tot_inj'] = v_dep_age['n_dose1'] + v_dep_age['n_complet']
    v_dep_age = v_dep_age[['jour', 'tot_inj']].copy().set_index('jour')


    return {
        'population'    : dep_row['pop'],
        'dep'           : dep,
        'nom_dep'       : f"{dep} - {dep_row['nom_dep']}",
        'age'           : age,
        'total_dose1'   : total_dose1,
        'total_complet' : total_complet,
        'injections'    : v_dep_age 
    }

def compute_agg(computed_vacc, by=None):
    '''
    Computes aggregated values for:
    - all deps by age
    - all ages by dep
    - all deps all ages
    '''

    if by == 'dep':
        agg = computed_vacc[['dep', 'nom_dep', 'age', 'population', 'total_dose1', 'total_complet', 'injections']].groupby('dep').apply(sum)
        agg = agg.drop(columns=['age'])
        agg['dep'] = agg['dep'].apply(lambda x: x[:len(x)//9])
        agg['nom_dep'] = agg['nom_dep'].apply(lambda x: x[:len(x)//9])
    elif by == 'age':
        agg = computed_vacc[['dep', 'nom_dep', 'age', 'population', 'total_dose1', 'total_complet', 'injections']].groupby('age').apply(sum)
        agg = agg.drop(columns=['age', 'dep'])
        agg['nom_dep'] = 'France'
    else:
        agg = pd.DataFrame(computed_vacc[['nom_dep', 'population', 'total_dose1', 'total_complet', 'injections']].sum()).T
        agg['nom_dep'] = 'France'

    return agg

def compute_avg(df):
    '''
    iterated over tables
    Add columns for percentages, and rolling avg of total injections for last 30 days
    '''
    df['pc_dose1'] = df['total_dose1'] / df['population']
    df['pc_complet'] = df['total_complet']  / df['population']

    mm_injections = df['injections'].rolling(7).mean()
    # keep last 30 days
    df['mm_injections'] = mm_injections.tail(30).copy()

    return df

def compute_all():
    '''
    computes all tables
    output dict, keys : 'all', 'by_dep', 'by_age', 'france'
    '''

    vacc, departements = load_format_data()

    computed_vacc = departements.apply(lambda x: compute_vacc(vacc, x), axis=1, result_type='expand')

    by_dep = compute_agg(computed_vacc, by='dep')
    by_age = compute_agg(computed_vacc, by='age')
    france  = compute_agg(computed_vacc)

    all = computed_vacc.apply(lambda x: compute_avg(x), axis=1, result_type='expand')
    by_dep = by_dep.apply(lambda x: compute_avg(x), axis=1, result_type='expand')
    by_age = by_age.apply(lambda x: compute_avg(x), axis=1, result_type='expand')
    france = france.apply(lambda x: compute_avg(x), axis=1, result_type='expand')

    return {
        'all'    : all, 
        'by_dep' : by_dep, 
        'by_age' : by_age, 
        'france' : france
    }

#%%
################
# bullet chart
################

def make_bullet(df_fr, df_dep=None, dep=None, dose=1):
    '''
    Input : dict of 2 dataframes (by dep, france) already filtered for the relevant age
    Returns a bullet chart go 
    '''     
    if dep:
        nom_dep = df_dep[df_dep['dep'] == dep]['nom_dep'].iloc[0]
        nom_dep = '<br>'.join(nom_dep)
       
        if dose==1:
            score, target = df_dep['pc_dose1'].iloc[0], df_fr['pc_dose1'].iloc[0]
        elif dose==2:
            score, target = df_dep['pc_complet'].iloc[0], df_fr['pc_complet'].iloc[0]
    else:
        nom_dep = 'France'
        if dose==1:
            score, target = df_fr['pc_dose1'].iloc[0], 1
        elif dose==2:
            score, target = df_fr['pc_complet'].iloc[0], 1

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
            ]
        },
        domain = {'x' : [0,1], 'y' : [0,1]}
    )
    if dose == 1:
        chart['title'] = {'text' : nom_dep, 'align' : 'left', 'font' : {'size' : 14}}
    return chart
# result = compute_all()
# fig = go.Figure()
# fig.add_trace(make_bullet(result['france'], result['by_dep'], dep='14', dose=1))
# fig.show()
#%%
##############
#Sparklines
##############

def make_sparkline(df):
    '''
    Returns a sparkline go
    '''
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

def make_card(df):
    try:
        last_week = df['tot_inj'].iloc[-1]
    except:
        last_week = df.iloc[0]['tot_inj'].iloc[-1]
    try:
        minus_1W = df['tot_inj'].iloc[-8]
    except:
        minus_1W = df.iloc[0]['tot_inj'].iloc[-8]

    card = go.Indicator(
        value = last_week,
        number = {'font' : {'size' : 14}}, #'valueformat' : '>,.0f'},
        delta = {'reference' : minus_1W, 'position' : 'bottom',
                'relative' : True, 'valueformat' : '.0%', 'font' : {'size' : 10}},
        mode = 'number+delta',
        align = 'right'

    )
    return card

###############
# Full table
###############

def make_table(input_data, age=None):
    '''
    data is the output of compute_all()
    '''

    if age:
        all = input_data['all']
        df_all_dep = all[all['age'] == age].sort_values(by='pc_complet').reset_index()
        df_france = input_data['by_age'].loc[[age]]
    else:
        df_all_dep = input_data['by_dep'].sort_values(by='pc_complet').reset_index()
        df_france = input_data['france']
    
    n_dep = df_all_dep.shape[0]
    n_rows =  n_dep * 3 + 3
    n_cols = 4

    row_specs = [[None               , None               , {'type' : 'xy', 'rowspan' : 3}, {'type' : 'domain', 'rowspan' : 3}],
                 [{'type' : 'domain'}, {'type' : 'domain'}, None                          , None                              ],
                 [None               , None               , None                          , None                              ]],
    
    specs = row_specs[0] * (n_dep + 1)
    row_heights = [1, 4.5, 1] * (n_dep + 1)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs = specs,
        column_widths = [2.5, 2.5, 1, 0.5],
        row_heights = row_heights,
        horizontal_spacing = 0.03,
        vertical_spacing   = 0.0,
        print_grid = False
    )

    fig.print_grid

    def make_row(fig, df_france, df_all_dep=None, idx=None, dep=None):
        '''
        '''

        if dep:
            df = df_all_dep[df_all_dep['dep'] == dep]
            # nom_dep = df['nom_dep'][0]
            # middle row in the department's row (1st row is for Fr, each dep's row takes 3 rows in the grid)
            mid_row = (idx  + 1) * 3 + 2
        else:
            df = df_france
            # nom_dep = df['nom_dep']
            mid_row = 2

        # dep_nom = textwrap.wrap(dep_nom, width=15)
        # dep_nom = '<br>'.join(dep_nom)

        fig.append_trace(
            make_bullet(df_france, df, dep, dose=1),
            mid_row, 1)

        # fig.update_traces(
        #     title = {'text' : dep_nom, 'align' : 'left', 'font' : {'size' : 14}}, 
        # )
        fig.append_trace(
            make_bullet(df_france, df, dep, dose=2),
            mid_row, 2
        )
        fig.append_trace(
            make_sparkline(df)[0],
            mid_row-1, 3
        )
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
        fig.add_trace(make_sparkline(df)[1], mid_row-1, 3)
        # min_inj = min(df['mm_injections'].iloc[0]['tot_inj']) - 100
        # max_inj = max(df['mm_injections'].iloc[0]['tot_inj']) + 100
        # fig.update_yaxes(range=[min_inj, max_inj], row=mid_row-1, col=3)
        fig.append_trace(make_card(df['mm_injections']), mid_row-1,4)

    # France total row
    make_row(fig, df_france)

    # departements rows
    for idx, dep in enumerate(df_all_dep['dep'].unique()):
        make_row(fig, df_france, df_all_dep, idx, dep)

    fig.update_layout(
        height= 50 * (n_rows + 1),
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
    for i in range(1,n_dep,2)
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
    return fig
#%%
result = compute_all()
#%%
fig = make_table(result, 49)
#%%
figdict = fig.to_dict()
# fig.show()

#%%
fig = go.Figure()

fig.add_trace(make_bullet(result['france'], dose=2))

fig.show()

#%%

fig = go.Figure()
fig.add_trace(make_bullet(result['france'], result['by_dep'], dep='75', dose=1))
fig.show()
#%%

fig = go.Figure()
fig.add_trace(make_sparkline(result['by_dep'].loc['14','mm_injections']['tot_inj']))
fig.show()
#%%
fig = go.figure()


