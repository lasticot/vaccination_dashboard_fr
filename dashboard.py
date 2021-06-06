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
    '29' : '18 - 29 ans', 
    '39' : '30 - 39 ans',
    '49' : '40 - 49 ans', 
    '59' : '50 - 59 ans', 
    '69' : '60 - 69 ans', 
    '79' : '70 - 79 ans', 
    '80' : '80 ans et plus'
}
# table format
n_cols = 9
fig_width = 19
width_ratios = [5, 9, 9, 5, 4, 5, 4, 5, 4]
wspace = 0.04

# Lookup des noms de départements
df_nom_dep = pd.read_excel('nom_dep.xlsx', engine='openpyxl', dtype={'dep':str}, usecols=['dep', 'nom_dep']).set_index('dep')
df_nom_dep = df_nom_dep['nom_dep'].copy()

#######
# chargement et formattage des data
#######

def load_compute_data():
    # vaccination
    # df1 = pd.read_csv('raw.csv', delimiter=';', 
    #     parse_dates=['jour'], dtype={'dep':str})
    df1 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/83cbbdb9-23cb-455e-8231-69fc25d58111', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})
    # les données pour la France (dep '00') sont vides dans le fichier par département (!!??), je remplace donc par les données du fichier France
    # df2 = pd.read_csv('vacc_fr.csv', delimiter=';', 
    #     parse_dates=['jour'], dtype={'dep':str})
    df2 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/54dd5f8d-1e2e-4ccb-8fb8-eac68245befd', delimiter=';', 
        parse_dates=['jour'], dtype={'dep':str})

    # données des cas détectés 
    # df3 = pd.read_csv('test.csv', sep=';', dtype={'dep':str}, infer_datetime_format=True, parse_dates=['jour'], 
    #                 header=0, names=['dep', 'jour', 'pos', 'test', 'clage', 'pop'])
    df3 = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675', sep=';', dtype={'dep':str}, infer_datetime_format=True, parse_dates=['jour'], 
                    header=0, names=['dep', 'jour', 'pos', 'test', 'clage', 'pop'])

    # changement de nom
    vacc = df1.copy()
    france = df2.copy()
    test = df3.copy()

    # on supprime les dep '00' '970' et '750' (??) du fichier départemental et 98 qui n'est pas dans fichier test
    # (les dep '00' qui sont censés contenir l'agg au niveau France sont vides donc remplcées par le fichier fra)
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

    # # extrapolation de la population des 20-29 pour correspondre à la fourchette 18-29 de vacc (*1.2)
    # test['pop'] = test.apply(lambda x: x['pop'] * 1.2 if x['clage'] == 29 else x['pop'], axis=1)

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
    
    # recalcul du % de couverture pour les nouvelles clages en utilisant la pop du fichier test
    # !!!!!!!!!!!!!! problème pour la clage 20-29 qui est 18-29 dans le fichier vacc
    vacc['couv_dose1'] = vacc['n_cum_dose1'] / vacc['pop']
    vacc['couv_complet'] = vacc['n_cum_complet'] / vacc['pop']
    # nb total d'injections
    vacc['inj'] = vacc['n_dose1'] + vacc['n_complet']
    # Nb de personnes non vaccinées estimée
    vacc['part_vacc'] = vacc['n_cum_dose1'] - vacc['n_cum_complet']
    # Nb de personnes partiellement vaccinées
    vacc['non_vacc'] = vacc['pop'] - vacc['n_cum_complet']
    # Nb d'inj à effectuer pour couvrir tout le monde (2 por les non_vacc, 1 pour les part_vacc)
    vacc['inj_todo'] = vacc['part_vacc'] + vacc['non_vacc'] * 2
    # Colonnes qui servent de placeholder les indicateurs sont calculés après le pivot
    vacc['ratio'] = 0
    vacc['nouveaux'] = 0
    vacc['pos_sem'] = 0
    vacc['inc_S'] = 0
    vacc['inc']   = 0

    # il faut unstack pour pouvoir calculer les moyennes et sommes mobiles
    # 3 levels pour les col : 
    #   - indicateur
    #   - clage
    #   - dep
    pivot = vacc.pivot(index='jour', columns=['clage', 'dep'], 
                values=['couv_dose1', 'couv_complet', 'n_dose1', 'n_complet', 'nouveaux', 'pop', 'ratio', 'pos', 'pos_sem', 'inj', 'inj_todo', 'non_vacc', 'inc_S', 'inc'])
    # Nb d'injections par semaine glissante
    pivot['inj'] = pivot['inj'].rolling(7).sum()
    # Ratio nb d'inj / non_vaccinés 1 semaine auparavant
    pivot['ratio'] = pivot['inj'].div(pivot['inj_todo'].shift(7)).replace(np.inf, np.nan) * 100
    # Ratio du nb d'injections dose1 / inj complet
    pivot['nouveaux'] = pivot['n_dose1'].rolling(7).sum().div(pivot['inj']).replace(np.inf, np.nan)
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

def make_bullet(ax, df, target=None, dose=1):
    '''
    returns ax
    '''
    score = df[-1]
    if dose == 1:
        bar_color = colors['bullet_bar_1dose']
    else:
        bar_color = colors['bullet_bar_complet']
    ax.set_aspect(0.014)
    ax.barh(0.5, 1, height=6, color=colors['bullet_bkg'], align='center')
    ax.barh(0.5, score,  height=3, color=bar_color, align='center')
    if target:
        ax.axvline(target, color='black', ymin=0.20, ymax=0.80)
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

def make_sparkline(ax, df, plus_good=True):
    ''' returns ax '''
    if plus_good:
        color = colors['sparkline']
    else:
        color  = colors['sparkline_neg']
    markers_on = [-1]
    ax.plot(df.index, df, '-o', color=color, markevery=markers_on)
    ax.fill_between(df.index, df.min(), df, color=color, 
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

def make_card(ax, df, plus_good=True, pourcentage=False):
    '''
    Displays the value of the metric for the last 30 days and the pc change against previous 30 days
    '''

    if plus_good:
        color_inc = colors['value+']
        color_dec = colors['value-']
    else:
        color_inc = colors['value-']
        color_dec = colors['value+']
    last = df.iloc[-1]
    minus_1W = df.iloc[-8]
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

    if pourcentage:
        value = f"{last:.0%}"
    else:
        value = f"{last:.0f}" 
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

def filter_sort_selection(df, dep='every', age=0):
    # calcul targets pour couv_dose1, couv_complet retournées dans un df
    #  - dep sélectionné : target niveau France pour chaque clage (différent pour chaque clage)
    #  - age sélectionné : target niveau France pour la clage sélectionnée (même target pour tous les dep)
    last_date = df.index[-1]
    targets = df.loc[last_date, (['couv_dose1', 'couv_complet'], slice(None), '00')].droplevel('dep')
    # Si un dep est sélectionné affiche toutes les classes d'âge pour ce dep (même si une clage a été sélectionnée)
    passed_dep = dep
    if passed_dep != 'every':
        age = slice(None)
        to_drop = 'dep'
        filtered = df.loc[:, (slice(None), age, dep)].droplevel(to_drop, axis=1)
    # si pas de dep sélectionné, affiche tous les dep pour la clage sélectionnée (0 par défaut)
    else:
        dep = slice(None)
        to_drop = 'clage'
        filtered = df.loc[:, (slice(None), age, slice(None))].droplevel(to_drop, axis=1)
    # tri des départements par couv_complet décroissant avec France en tête et COM à la fin
    # retourne un df avec pour col : 
    #   - level 0 : indicateur
    #   - level 1 : clage
    if passed_dep != 'every':
        # tri par défaut des clages
        order = filtered['couv_complet'].columns
    else:
    # départements triés par % de couverture complète décroissant avec France en tête et COM à la fin
        COM = ['971', '972', '973', '974', '975', '976', '977', '978']
        order = filtered.loc[last_date, 'couv_complet'].sort_values(ascending=False)
        order = pd.concat([order[order.index=='00'], order[(~order.index.isin(COM)) & (order.index != '00')], order[order.index.isin(COM)]])
        order  = order.index
    return {'df': filtered, 'targets': targets, 'sorting': order, 'last_date': last_date}

def make_table_header(dep='every', age=0):
    global df_nom_dep, clages, colors, fig_width, n_cols, width_ratios, wspace
    if dep != 'every':
        selected = df_nom_dep[dep]
    else:
        selected = clages[str(age)]
    header_fig = plt.figure(figsize=(fig_width, 1.3), facecolor=colors['bullet_bar_complet'])
    header_div = header_fig.add_gridspec(2, n_cols, hspace=1.5, wspace=wspace, width_ratios=width_ratios)
    header_nom_dep = header_fig.add_subplot(header_div[0:1,0])
    make_header(header_nom_dep, "")
    header_bullet_top = header_fig.add_subplot(header_div[0,1:3])
    make_header(header_bullet_top, f"Pourcentage de la population de {selected}...", width=60, fontsize=14, fontcolor=colors['header_font'])

    header_bullet_left = header_fig.add_subplot(header_div[1,1])
    make_header(header_bullet_left, "...partiellement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])
    header_bullet_right = header_fig.add_subplot(header_div[1,2])
    make_header(header_bullet_right, "...entièrement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])

    header_nouv_top = header_fig.add_subplot(header_div[0,3:5])
    make_header(header_nouv_top, "% des primo-vaccinations sur le total des inj. 7jrs glissants", fontsize=12, width=26, fontcolor=colors['header_font'])
    header_sparkline_left = header_fig.add_subplot(header_div[1,3])
    make_header(header_sparkline_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_sparkline_right = header_fig.add_subplot(header_div[1,4])
    make_header(header_sparkline_right, "7 dern. jrs \n (% p.r 7 jrs préc.)", fontsize=11, width = 13, fontcolor=colors['header_font'])

    header_sparkline_top = header_fig.add_subplot(header_div[0,5:7])
    make_header(header_sparkline_top, "Nb d'inj. pour 100 pers. non vaccinées 7 jrs glissants", fontsize=12, width=26, fontcolor=colors['header_font'])
    header_sparkline_left = header_fig.add_subplot(header_div[1,5])
    make_header(header_sparkline_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_sparkline_right = header_fig.add_subplot(header_div[1,6])
    make_header(header_sparkline_right, "7 dern. jrs \n (% p.r 7 jrs préc.)", fontsize=11, width = 13, fontcolor=colors['header_font'])

    header_inc_top = header_fig.add_subplot(header_div[0,7:])
    make_header(header_inc_top, "Incidence parmi les pers non vaccinées moy. glissante 7 jrs", fontsize=12, width=26, fontcolor=colors['header_font'])
    header_inc_left = header_fig.add_subplot(header_div[1,7])
    make_header(header_inc_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_inc_right = header_fig.add_subplot(header_div[1,8])
    make_header(header_inc_right, "7 dern. jrs \n (% p.r 7 jrs préc.)", fontsize=11, width = 13, fontcolor=colors['header_font'])
    return header_fig

def make_table(data, dep='every', age=None):
    global clages, df_nom_dep
    df = data['df']
    targets = data['targets']
    sorting = data['sorting']
    if dep != 'every':
        lookup = clages
    else:
        lookup = df_nom_dep

    # table format
    n_rows =  len(sorting)
    fig_height = n_rows * 0.8
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=colors['background'])
    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.05, wspace=wspace, width_ratios=width_ratios)
    
    for idx, sub in enumerate(sorting):
        # sub est le dep ou la clage selon ce qui est entré dans la fonction
        # si un dep est sélectionné, sorting contient les clages qu'on veut afficher
        # sinon, l'âge est celui qui est en argument de make_table()
        if dep != 'every':
            age = sub
        # col 1 : nom du dep ou de la clage
        nom = lookup[str(sub)]
        ax_nom_dep = fig.add_subplot(grid[idx, 0])
        make_header(ax_nom_dep, nom, fontsize=14, halign='right')

        # col 2 : couv dose 1
        div_dose1 = grid[idx, 1].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose1 = fig.add_subplot(div_dose1[0,0])
        make_bullet(ax_dose1, df.loc[:, ('couv_dose1', sub)], targets.loc['couv_dose1', age])

        # col 3 : couv complet
        div_dose2 = grid[idx, 2].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose2 = fig.add_subplot(div_dose2[0,0])
        make_bullet(ax_dose2, df.loc[:, ('couv_complet', sub)], targets.loc['couv_complet', age], dose=2)

        ax_spark_nouv =  fig.add_subplot(grid[idx, 3])
        make_sparkline(ax_spark_nouv, df.loc[:, ('nouveaux', sub)])
        
        ax_card_nouv = fig.add_subplot(grid[idx, 4])
        make_card(ax_card_nouv, df.loc[:, ('nouveaux', sub)], pourcentage=True)

        ax_spark =  fig.add_subplot(grid[idx, 5])
        make_sparkline(ax_spark, df.loc[:, ('ratio', sub)])
        
        ax_card = fig.add_subplot(grid[idx, 6])
        make_card(ax_card, df.loc[:, ('ratio', sub)])

        ax_spark_inc =  fig.add_subplot(grid[idx, 7])
        make_sparkline(ax_spark_inc, df.loc[:, ('inc_S', sub)], plus_good=False)
        
        ax_card_inc = fig.add_subplot(grid[idx, 8])
        make_card(ax_card_inc, df.loc[:, ('inc_S', sub)], plus_good=False)
        # plt.subplots_adjust(left=0, right=0.9)
        plt.xticks([])
        plt.yticks([])
    return fig    



# %%
