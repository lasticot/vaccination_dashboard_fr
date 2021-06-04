#%%
from collections import OrderedDict
from datetime import datetime, timedelta
from os import rename
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    'background'          : 'whitesmoke', 
    'bullet_bkg'          : 'white',
    'bullet_bar_complet'  : '#386CB9',
    'bullet_bar_1dose'    : '#7299D5',
    'sparkline'           : '#386CB9',
    'sparkline_neg'       : 'purple', 
    'value+'              : 'darkgreen', 
    'value-'              : 'darkred',
    'header'              : '#386CB9', 
    'header_font'         : 'white',
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

def load_compute_data():
    # vaccination
    df1 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/83cbbdb9-23cb-455e-8231-69fc25d58111', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})

    df2 = pd.read_excel('nom_dep.xlsx', engine='openpyxl', dtype={'dep':str})

    # les données pour la France (dep '00') sont vides dans le fichier par département (!!??), je remplace donc par les données du fichier France
    df3 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/54dd5f8d-1e2e-4ccb-8fb8-eac68245befd', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})

    # données des cas détectés 
    df4 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675', sep=';', dtype={'dep':str}, infer_datetime_format=True, parse_dates=['jour'], 
                    header=0, names=['dep', 'jour', 'pos', 'test', 'clage', 'pop'])

    # changement de nom
    vacc = df1.copy()
    departements = df2.copy()
    france = df3.copy()
    test = df4.copy()

    # on supprime les dep '00' '970' et '750' (??) du fichier départemental et 98 qui n'est pas dans fichier test
    # (les dep '00' qui sont censés contenir l'agg au niveau France sont vides donc remplcées par le fichier fra)
    # !!! plutôt sélectionner les dep à inclure ? !!!!!!!!
    excluded = ['00', '750', '98']
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

    vacc = vacc.merge(test[['dep', 'clage', 'jour', 'pos', 'test', 'pop']], on=['dep', 'clage', 'jour'], suffixes=(None, "_test"))
    
    # recalcul du % de couverture en utilisant la pop du fichier test
    vacc['couv_dose1'] = vacc['n_cum_dose1'] / vacc['pop']
    vacc['couv_complet'] = vacc['n_cum_complet'] / vacc['pop']
    # nb total d'injections, somme mobile 7 derniers jours
    vacc['inj'] = vacc['n_dose1'] + vacc['n_complet']
    # Nb de personnes non vaccinées estimée, somme mobile 7 derniers jours
    vacc['non_vacc'] = vacc['pop'] - vacc['n_cum_complet']
    # ratio du nombre d'injections sur la part des personnes restant à vacciner
    # division par pop au lieu de non_vacc pour éviter div par 0
    # Colonnes qui servent de placeholder les indicateurs sont calculés après le pivot
    vacc['ratio'] = 0
    vacc['pos_sem'] = 0
    vacc['inc_S'] = 0
    vacc['inc']   = 0

    # il faut unstack pour pouvoir calculer les moyennes et sommes mobiles
    # 3 levels pour les col : 
    #   - indicateur
    #   - clage
    #   - dep
    pivot = vacc.pivot(index='jour', columns=['clage', 'dep'], 
                values=['couv_dose1', 'couv_complet', 'pop', 'ratio', 'pos', 'pos_sem', 'inj', 'non_vacc', 'inc_S', 'inc'])
    # Nb d'injections par semaine glissante
    pivot['inj'] = pivot['inj'].rolling(7).sum()
    # Ratio nb d'inj / non_vaccinés 1 semaine auparavant
    pivot['ratio'] = pivot['inj'].div(pivot['non_vacc'].shift(7)).replace(np.inf, np.nan)
    # Nombre de nouveaux cas par semaine glissante
    pivot['pos_sem'] = pivot['pos'].rolling(7).sum()
    # Taux d'incidence calculée sur la population non vaccinée 1 semaine auparavant
    pivot['inc_S'] = pivot['pos_sem'].div(pivot['non_vacc'].shift(7)).replace(np.inf, np.nan) * 100_000
    # Taux d'incidence calculée sur la population totale
    pivot['inc'] = pivot['pos_sem'].div(pivot['pop']).replace(np.inf, np.nan) * 100_000

    # vacc = vacc.merge(departements, how='left', on='dep')
    # vacc['nom_dep'] = vacc.dep.str.cat(vacc.nom_dep, sep=' - ')
    # # Remplacer nom_dep NaN par 'France'
    # vacc['nom_dep'].replace(np.nan, 'France', inplace=True)

    return pivot.tail(30)

#%%
################
# bullet chart
################

def make_bullet(ax, df, dep, target=None, dose=1):

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
    if dep != '00':
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

def make_sparkline(ax, df, metric, plus_good=True):
    if plus_good:
        color = colors['sparkline']
    else:
        color  = colors['sparkline_neg']
    markers_on = [-1]
    ax.plot(df.index, df[metric], '-o', color=color, markevery=markers_on)
    ax.fill_between(df.index, df[metric].min(), df[metric], color=color, 
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

def make_card(ax, df, metric, plus_good=True):
    '''
    Displays the value of the metric for the last 30 days and the pc change against previous 30 days
    '''

    if plus_good:
        color_inc = colors['value+']
        color_dec = colors['value-']
    else:
        color_inc = colors['value-']
        color_dec = colors['value+']
    last = df[metric].iloc[-1]
    minus_1W = df[metric].iloc[-8]
    # except:
    #     minus_1W = df.iloc[0]['tot_inj'].iloc[-8]
    if minus_1W != 0:
        pc = (last - minus_1W) / minus_1W
    else:
        pc = np.nan
    if pc > 0:
        color_pc = color_inc
    else:
        color_pc = color_dec

    value = f"{last:.0f}" # .format(last * 1000) # human_format(last, k=True)
    pc = "({:+.0%})".format(pc)
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

    # Lookup des noms de départements
    df_nom_dep = pd.read_excel('nom_dep.xlsx', engine='openpyxl', dtype={'dep':str}).set_index('dep')

    df_age = df.loc[:, (slice(None), age, slice(None))].copy()
    # départements triés par % de couverture complète décroissant avec France en tête et COM à la fin
    couv_complet = pd.DataFrame(df_age['couv_complet'][age].iloc[-1].sort_values(ascending=False).rename('couv'))
    sort_france = couv_complet.loc[['00']]
    com = ['971', '972', '973', '974', '975', '976', '977', '978']
    sort_com = couv_complet.loc[com].sort_values('couv', ascending=False)
    sort_all = couv_complet[(couv_complet.index != '00') & (~couv_complet.index.isin(com))].sort_values('couv', ascending=False)
    sorted = pd.concat([sort_france, sort_all, sort_com])

    target_dose1 = df_age['couv_dose1'][age].iloc[-1]['00']
    target_complet = couv_complet.loc['00', 'couv']

    # table format
    n_dep = len(sorted.index.unique())
    n_rows =  n_dep
    n_cols = 7
    fig_width = 18.5
    fig_height = n_rows * 0.8
    width_ratios = [2, 5, 5, 2, 1.5, 2, 1.5]
    wspace = 0.04

    # header figure
    header_fig = plt.figure(figsize=(fig_width, 1.3), facecolor=colors['bullet_bar_complet'])
    header_div = header_fig.add_gridspec(2, n_cols, hspace=1.5, wspace=wspace, width_ratios=width_ratios)
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
    header_sparkline_top = header_fig.add_subplot(header_div[0,3:5])
    make_header(header_sparkline_top, "Nb d'inj. 7 jrs glissants pour 100 pers. non vaccinées", fontsize=12, width=26, fontcolor=colors['header_font'])
    header_sparkline_left = header_fig.add_subplot(header_div[1,3])
    make_header(header_sparkline_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_sparkline_right = header_fig.add_subplot(header_div[1,4])
    make_header(header_sparkline_right, "7 dern. jrs \n (% p.r 7 jrs préc.)", fontsize=11, width = 13, fontcolor=colors['header_font'])

    header_inc_top = header_fig.add_subplot(header_div[0,5:])
    make_header(header_inc_top, "Incidence parmi les pers non vaccinées moy. glissante 7 jrs", fontsize=12, width=26, fontcolor=colors['header_font'])
    header_inc_left = header_fig.add_subplot(header_div[1,5])
    make_header(header_inc_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_inc_right = header_fig.add_subplot(header_div[1,6])
    make_header(header_inc_right, "7 dern. jrs \n (% p.r 7 jrs préc.)", fontsize=11, width = 13, fontcolor=colors['header_font'])
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=colors['background'])

    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.05, wspace=wspace, width_ratios=width_ratios)
    
    for idx, dep in enumerate(sorted.index):
        df_dep = df_age.xs((age, dep), level=('clage', 'dep'), axis=1)
        nom_dep = df_nom_dep.loc[dep, 'nom_dep']
        ax_nom_dep = fig.add_subplot(grid[idx, 0])
        make_header(ax_nom_dep, nom_dep, fontsize=14, halign='right')

        div_dose1 = grid[idx, 1].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose1 = fig.add_subplot(div_dose1[0,0])
        make_bullet(ax_dose1, df_dep, dep, target_dose1, dose=1)

        div_dose2 = grid[idx, 2].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose2 = fig.add_subplot(div_dose2[0,0])
        make_bullet(ax_dose2, df_dep, dep, target_complet, dose=2)

        ax_spark =  fig.add_subplot(grid[idx, 3])
        make_sparkline(ax_spark, df_dep, 'ratio')
        
        ax_card = fig.add_subplot(grid[idx, 4])
        make_card(ax_card, df_dep * 100, 'ratio')

        ax_spark_inc =  fig.add_subplot(grid[idx, 5])
        make_sparkline(ax_spark_inc, df_dep, 'inc_S', plus_good=False)
        
        ax_card_inc = fig.add_subplot(grid[idx, 6])
        make_card(ax_card_inc, df_dep, 'inc_S', plus_good=False)
        # plt.subplots_adjust(left=0, right=0.9)
        plt.xticks([])
        plt.yticks([])
    
    return header_fig, fig;


