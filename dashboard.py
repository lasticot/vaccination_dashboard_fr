#%%
from collections import OrderedDict
from datetime import datetime, timedelta
from os import rename
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    'background' : 'whitesmoke', 
    'bullet_bkg' : 'white',
    'bullet_bar_complet': '#386CB9',
    'bullet_bar_1dose': '#7299D5',
    'sparkline' : '#386CB9',
    'value+'    : 'darkgreen', 
    'value-'    : 'darkred',
    'header'    : '#386CB9', 
    'header_font' : 'white'
}

clages = {
    '0'  : '18 ans et plus',
    '24' : '18 - 24 ans',
    '29' : '24 - 29 ans', 
    '39' : '30 - 39 ans',
    '49' : '40 - 49 ans', 
    '59' : '50 - 59 ans', 
    '64' : '60 - 64 ans', 
    '69' : '65 - 69 ans', 
    '74' : '70 - 74 ans', 
    '79' : '75 - 79 ans', 
    '80' : '80 ans et plus'
}
#######
# chargement et formattage des data
#######

def load_format_data():
    # vaccination
    df1 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/83cbbdb9-23cb-455e-8231-69fc25d58111', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})

    df2 = pd.read_excel('nom_dep.xlsx', engine='openpyxl', dtype={'dep':str})

    # les données pour la France (dep '00') sont vides dans le fichier par département (!!??), je remplace donc par les données du fichier France
    df3 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/54dd5f8d-1e2e-4ccb-8fb8-eac68245befd', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})

    # données des cas détectés 
    df4 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675', sep=';', dtype={'dep':str}, infer_datetime_format=True, parse_dates=['jour'])

    # changement de nom
    vacc = df1.copy()
    departements = df2.copy()
    france = df3.copy()
    test = df4.copy()

    # on supprime les dep '00' '970' et '750' (??) du fichier départemental et 98 qui n'est pas dans fichier test
    # (les dep '00' qui sont censés contenir l'agg au niveau France sont vides donc remplcées par le fichier fra)
    excluded = ['00', '750', '970', 98]
    vacc = vacc[~vacc.dep.isin(excluded)].copy()
    # je remplace la colonne fra du fichier France par une colonne dep avec '00'
    france = france.rename(columns={'fra': 'dep'})
    france['dep'] = '00'
    vacc = pd.concat([vacc, france], ignore_index=True)

    # on garde les 37 derniers jours
    last_date = min(max(vacc.jour), max(test.jour))
    first_date = last_date - timedelta(days=37)
    vacc = vacc[(first_date <= vacc.jour) & (vacc.jour <= last_date)]
    test = test[(first_date <= test.jour) & (test.jour <= last_date)]

    # harmonisation des classes d'âges
    # - fichier vaccin commence à 18 ans, fichier incidence commence à 19 ans
    # - ajouter âge 0 dans incidence pour l'aggrégation
    # - dans vaccination fusionner les classes d'âge 24-29, 64-69, 74-79
    # - dans incidence fusionner les classes d'âge 80-90 et supprimer les âges 0, 9, 19
    def new_age_vacc(age):
        if age == 24:
            return 29
        elif age == 64:
            return 69
        elif age == 74:
            return 79
        else:
            return age
    def new_age_test(age):
        if age in [89, 90]:
            return 80
        else:
            return age

    # Remplace clage puis groupby dans vacc 
    vacc.rename(columns={'clage_vacsi':'clage'}, inplace=True)
    vacc['clage'] = vacc.clage.apply(new_age_vacc)
    vacc = vacc.groupby(['dep', 'clage',  'jour'], as_index=False).sum()

    # suppression clage 0, 9, 19, remplace clages puis groupby
    test.rename(columns={'cl_age90':'clage'}, inplace=True)
    test = test[test.clage >= 29]
    test['clage'] = test.clage.apply(new_age_test)
    test = test.groupby(['dep', 'clage', 'jour'], as_index=False).sum()

    # dans test ajouter une classe d'âge 0 aggrégeant toutes les clages
    test_clage_agg = test.groupby(['dep', 'jour'], as_index=False).sum()
    test_clage_agg['clage'] = 0
    test = pd.concat([test, test_clage_agg], ignore_index=True)

    # dans test ajouter un dep '00' aggrégeant au niveau france
    test_fr_agg = test.groupby(['clage', 'jour'], as_index=False).sum()
    test_fr_agg['dep'] = '00'
    test = pd.concat([test, test_fr_agg], ignore_index=True)

    vacc = vacc.merge(test[['dep', 'clage', 'jour', 'P', 'T', 'pop']], on=['dep', 'clage', 'jour'], suffixes=(None, "_test"))
    
    # recalcul du % de couverture en utilisant la pop du fichier test
    vacc['couv_dose1'] = vacc['n_cum_dose1'] / vacc['pop']
    vacc['couv_complet'] = vacc['n_cum_complet'] / vacc['pop']
    # nb total d'injections, somme mobile 7 derniers jours
    vacc['inj'] = vacc['n_dose1'] + vacc['n_complet']
    # Nb de personnes non vaccinées estimée, somme mobile 7 derniers jours
    vacc['non_vacc'] = vacc['pop'] - vacc['n_cum_complet']
    # ratio du nombre d'injections sur la part des personnes restant à vacciner
    # division par pop au lieu de non_vacc pour éviter div par 0
    # la colonne ratio n'est qu'un placeholder. Il est recalculé après le pivot
    vacc['ratio'] = vacc['inj'].div(vacc['pop']).replace(np.inf, np.nan)
    # calcul de l'incidence pour 100_000 habitants
    vacc['inc'] = vacc['P'].div(vacc['pop']).replace(np.inf, np.nan) * 100_000

    # il faut unstack pour pouvoir calculer les moyennes et sommes mobiles
    # 3 levels pour les col : 
    #   - indicateur
    #   - clage
    #   - dep
    pivot = vacc.pivot(index='jour', columns=['clage', 'dep'], values=['couv_dose1', 'couv_complet', 'ratio', 'inj', 'non_vacc', 'inc'])
    # calcul du ratio nb d'inj / non_vaccinés somme mobile 7j
    pivot['ratio'] = pivot['inj'].div(pivot['non_vacc']).replace(np.inf, np.nan).rolling(7).sum()
    # calcul de la mm 7j de l'incidence
    pivot['inc'] = pivot['inc'].rolling(7).mean()

    # vacc = vacc.merge(departements, how='left', on='dep')
    # vacc['nom_dep'] = vacc.dep.str.cat(vacc.nom_dep, sep=' - ')
    # # Remplacer nom_dep NaN par 'France'
    # vacc['nom_dep'].replace(np.nan, 'France', inplace=True)

    return pivot
#%%
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
    v_dep_age = vacc[(vacc['dep'] == dep) & (vacc['clage'] == age)].copy()

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

def make_bullet(ax, df, target=None, dose=1):

    if dose == 1:
        score = df['couv_dose1'].iloc[-1]
    elif dose == 2:
        score = df['couv_complet'].iloc[-1]

    if dose == 1:
        bar_color = colors['bullet_bar_1dose']
    if dose == 2:
        bar_color = colors['bullet_bar_complet']

    ax.set_aspect(0.015)
    ax.barh(0.5, 1, height=6, color=colors['bullet_bkg'], align='center')
    ax.barh(0.5, score,  height=3, color=bar_color, align='center')
    if df.dep.iloc[0] != '00':
        ax.axvline(target, color='black', ymin=0.15, ymax=0.85)
    ax.set_xlim([0,1])
    ax.set_facecolor(color='lightblue')
    plt.xticks([])
    plt.yticks([])

    ax.text(x=1.05, y=0.5, s=f'{score:.0%}', fontsize=14)
    return ax


#%%
######################
# SPARKLINE
#####################

def make_sparkline(ax, df):
    markers_on = [-1]
    # on graph les 30 derniers jours
    df_30 = df.tail(30)
    ax.plot(df_30.jour, df_30['ratio'], '-o', color=colors['sparkline'], markevery=markers_on)
    ax.fill_between(df_30.jour, df_30.ratio.min(), df_30.ratio, color=colors['sparkline'], 
        alpha = 0.2)
    ax.axis('off')
    ax.margins(y=0.6)
    return ax


#%%
################
# Card
################

def human_format(num, k=False):
    num = float('{:.0f}'.format(num))
    if k:
        magnitude = 0
        while abs(num) >= 10000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:.0f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    else:
        return "{}".format('{}'.format(num).rstrip('0').rstrip('.'))

def make_card(ax, df):
    '''
    Displays the value of the metric for the last 30 days and the pc change against previous 30 days
    '''
    last = df['ratio'].iloc[-1]
    minus_1W = df['ratio'].iloc[-8]
    # except:
    #     minus_1W = df.iloc[0]['tot_inj'].iloc[-8]
    if minus_1W != 0:
        pc = (last - minus_1W) / minus_1W
    if pc > 0:
        color_pc = colors['value+']
    else:
        color_pc = colors['value-']

    value = f"{last * 1000:.2f}" + u"\u2030" # .format(last * 1000) # human_format(last, k=True)
    pc = "{:+.2%}".format(pc)
    ax.text(x=0.5, y=0.5, s=value,  fontsize=14, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=0.5, s=pc, color=color_pc, fontsize = 10, ha='center', va='top', transform=ax.transAxes)
    ax.axis('off')
    ax.margins(x=0.3, y=0.4)
    return ax


#%%
########################
# header
########################

def make_header(ax, text, halign='center', width=15, fontsize=16, fontcolor='black'):
    if width == None:
        text_wrapped = text
    else:
        text_wrapped = textwrap.wrap(text, width = width)
        text_wrapped = '\n'.join(text_wrapped)
    if halign == 'right':
        xref = 1
    elif halign == 'left':
        xref = 0
    else:
        xref = 0.5
    ax.text(x=xref, y=0.5, s=text_wrapped, fontsize=fontsize, ha=halign, va='center', transform=ax.transAxes, color=fontcolor)
    ax.axis('off')
    return ax


#%%

###############
# Full table
###############

def make_table(df, age=0):

    df_age = df[df.clage == age].copy()
    # départements triés par % de couverture complète décroissant avec France en tête et COM à la fin
    df_france = df_age[df_age.dep == '00']
    df_france = df_france[df_france.jour == max(df_france.jour)].copy()
    exclude = df_age.dep.isin(['970', '971', '972', '973', '974', '975', '976', '977', '978', '98'])
    df_exclude = df_age[exclude]
    df_exclude_sorted = df_exclude[df_exclude.jour == max(df_exclude.jour)].sort_values(by='couv_complet', ascending=False)
    include = ~df_age.dep.isin(['00', '970', '971', '972', '973', '974', '975', '976', '977', '978', '98'])
    df_include = df_age[include]
    df_include_sorted = df_include[df_include.jour == max(df_include.jour)].sort_values(by='couv_complet', ascending=False)
    sorted = pd.concat([df_france, df_include_sorted, df_exclude_sorted]).reset_index()

    target_dose1 = df_france['couv_dose1'].iloc[-1]
    target_complet = df_france['couv_complet'].iloc[-1]

    # header figure
    header_fig = plt.figure(figsize=(15, 1.3), facecolor=colors['bullet_bar_complet'])
    width_ratios = [2, 5, 5, 2, 1.5]
    header_div = header_fig.add_gridspec(2, 5, hspace=0.05, width_ratios=width_ratios)
    header_nom_dep = header_fig.add_subplot(header_div[0:1,0])
    make_header(header_nom_dep, "")
    header_bullet_top = header_fig.add_subplot(header_div[0,1:3])
    if age == 0:
        make_header(header_bullet_top, f"Pourcentage de la population...", width=60, fontsize=14, fontcolor=colors['header_font'])
    else:
        clage = clages[str(age)]
        make_header(header_bullet_top, f"Pourcentage de la classe d'âge {clage}...", width=60, fontsize=14, fontcolor=colors['header_font'])

    header_bullet_left = header_fig.add_subplot(header_div[1,1])
    make_header(header_bullet_left, "...partiellement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])
    header_bullet_right = header_fig.add_subplot(header_div[1,2])
    make_header(header_bullet_right, "...entièrement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])
    header_sparkline_top = header_fig.add_subplot(header_div[0,3:])
    make_header(header_sparkline_top, "Nb d'inj. hebdo pour 1000 habitants non vaccinés", fontsize=12, width=22, fontcolor=colors['header_font'])
    header_sparkline_left = header_fig.add_subplot(header_div[1,3])
    make_header(header_sparkline_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_sparkline_right = header_fig.add_subplot(header_div[1,4])
    make_header(header_sparkline_right, "7 dern. jrs \n % p.r 7 jrs préc.", fontsize=11, width = 13, fontcolor=colors['header_font'])
    
    n_dep = len(sorted.dep.unique())
    n_rows =  n_dep
    n_cols = 5

    fig_width = 15
    fig_height = n_rows * 0.8
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=colors['background'])

    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.05, width_ratios=width_ratios)
    
    for idx, dep in enumerate(sorted.dep):
        df_dep = df_age[df_age.dep == dep].copy()
        nom_dep = df_dep['nom_dep'].iloc[0]
        ax_nom_dep = fig.add_subplot(grid[idx, 0])
        make_header(ax_nom_dep, nom_dep, fontsize=14, halign='right')

        div_dose1 = grid[idx, 1].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose1 = fig.add_subplot(div_dose1[0,0])
        make_bullet(ax_dose1, df_dep, target_dose1, dose=1)

        div_dose2 = grid[idx, 2].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose2 = fig.add_subplot(div_dose2[0,0])
        make_bullet(ax_dose2, df_dep, target_complet, dose=2)

        ax_spark =  fig.add_subplot(grid[idx, 3])
        make_sparkline(ax_spark, df_dep)
        
        ax_card = fig.add_subplot(grid[idx, 4])
        make_card(ax_card, df_dep)

        # plt.subplots_adjust(left=0, right=0.9)
        plt.xticks([])
        plt.yticks([])
    
    return header_fig, fig;
