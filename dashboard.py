#%%
from collections import OrderedDict
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
    '29' : '24 - 29 ans', 
    '39' : '30 - 39 ans',
    '49' : '40 - 49 ans', 
    '59' : '50 - 59 ans', 
    '64' : '60 - 64 ans', 
    '69' : '65 - 69 ans', 
    '74' : '70 - 74 ans', 
    '79' : '75 - 79 ans', 
    '80' : '80 ans +'
}
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

def make_bullet(ax, df_fr, df_dep=None, dep=None, dose=1):
    if dep:
        if dose==1:
            score, target = df_dep['pc_dose1'].iloc[0], df_fr['pc_dose1'].iloc[0]
        elif dose==2:
            score, target = df_dep['pc_complet'].iloc[0], df_fr['pc_complet'].iloc[0]
    else:
        if dose==1:
            score, target = df_fr['pc_dose1'].iloc[0], 1
        elif dose==2:
            score, target = df_fr['pc_complet'].iloc[0], 1

    if dose == 1:
        bar_color = colors['bullet_bar_1dose']
    if dose == 2:
        bar_color = colors['bullet_bar_complet']

    ax.set_aspect(0.015)
    ax.barh(0.5, 1, height=6, color=colors['bullet_bkg'], align='center')
    ax.barh(0.5, score,  height=3, color=bar_color, align='center')
    if dep:
        ax.axvline(target, color='black', ymin=0.15, ymax=0.85)
    ax.set_xlim([0,1])
    ax.set_facecolor(color='lightblue')
    plt.xticks([])
    plt.yticks([])

    ax.text(x=1.05, y=0.5, s=f'{score:.0%}', fontsize=14)

    # ax.margins(x=0.4)

    return ax


#%%
######################
# SPARKLINE
#####################

def make_sparkline(ax, df):
    markers_on = [-1]
    ax.plot(df, '-o', color=colors['sparkline'], markevery=markers_on)
    ax.fill_between(df.index, df.min(), df.iloc[:,0], color=colors['sparkline'], 
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
    last_week = df['tot_inj'][-1]
    # except:
    #     last_week = df.iloc[0]['tot_inj'].iloc[-1]
    minus_1W = df['tot_inj'].iloc[-8]
    # except:
    #     minus_1W = df.iloc[0]['tot_inj'].iloc[-8]
    if minus_1W != 0:
        pc = (last_week - minus_1W) / minus_1W
    if pc > 0:
        color_pc = colors['value+']
    else:
        color_pc = colors['value-']

    value = human_format(last_week, k=True)
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

def make_table(input_data, age=None):
    if age:
        all = input_data['all']
        df_all_dep = all[all['age'] == age].sort_values(by='pc_complet', ascending=False).reset_index()
        df_france = input_data['by_age'].loc[[age]]
    else:
        df_all_dep = input_data['by_dep'].sort_values(by='pc_complet', ascending=False).reset_index(drop=True)
        df_france = input_data['france']

    # header figure
    header_fig = plt.figure(figsize=(15, 1.3), facecolor=colors['bullet_bar_complet'])

    header_div = header_fig.add_gridspec(2, 5, hspace=0.05, width_ratios=[2, 5, 5, 2, 1.2])
    header_nom_dep = header_fig.add_subplot(header_div[0:1,0])
    make_header(header_nom_dep, "")
    header_bullet_top = header_fig.add_subplot(header_div[0,1:3])
    if age == None:
        make_header(header_bullet_top, f"Pourcentage de la population âgée de 24 ans et plus...", width=60, fontsize=14, fontcolor=colors['header_font'])
    else:
        clage = clages[str(age)]
        make_header(header_bullet_top, f"Pourcentage de la classe d'âge {clage}...", width=60, fontsize=14, fontcolor=colors['header_font'])

    header_bullet_left = header_fig.add_subplot(header_div[1,1])
    make_header(header_bullet_left, "...partiellement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])
    header_bullet_right = header_fig.add_subplot(header_div[1,2])
    make_header(header_bullet_right, "...entièrement vaccinée", fontsize=14, width=30, fontcolor=colors['header_font'])
    header_sparkline_top = header_fig.add_subplot(header_div[0,3:])
    make_header(header_sparkline_top, "Nb d'injections moy. mobile 7jrs", fontsize=12, width=18, fontcolor=colors['header_font'])
    header_sparkline_left = header_fig.add_subplot(header_div[1,3])
    make_header(header_sparkline_left, "30 derniers jrs", fontsize=11, fontcolor=colors['header_font'])
    header_sparkline_right = header_fig.add_subplot(header_div[1,4])
    make_header(header_sparkline_right, "7 dern. jrs \n % p.r 7 jrs préc.", fontsize=11, width = 13, fontcolor=colors['header_font'])
    
    n_dep = df_all_dep.shape[0]
    n_rows =  n_dep + 1
    n_cols = 4

    fig_width = 15
    fig_height = n_rows * 0.8
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=colors['background'])

    grid = fig.add_gridspec(n_rows, 5, hspace=0.05, width_ratios=[2, 5, 5, 2, 1.2])
    
    # france row
    nom_dep = 'France'
    ax_fr = fig.add_subplot(grid[0, 0])
    make_header(ax_fr, 'France', halign='right')
    # plt.xticks([])
    # plt.yticks([])

    # ajout d'un gridspec vide pour bon alignement du pourcentage
    div_dose1 = grid[0, 1].subgridspec(1, 2, width_ratios=[6,1])
    ax_fr_dose1 = fig.add_subplot(div_dose1[0,0])
    make_bullet(ax_fr_dose1, df_france, df_france, dose=1)

    div_dose2 = grid[0, 2].subgridspec(1, 2, width_ratios=[6,1])
    ax_fr_dose2 = fig.add_subplot(div_dose2[0,0])
    make_bullet(ax_fr_dose2, df_france, df_france, dose=2)

    df_inj_fr = df_france['mm_injections'].iloc[0]
    ax_fr_spark =  fig.add_subplot(grid[0, 3])
    make_sparkline(ax_fr_spark, df_inj_fr)
    
    ax_fr_card = fig.add_subplot(grid[0, 4])
    make_card(ax_fr_card, df_inj_fr)

    for row in range(0,n_rows-1):
        nom_dep 
        df_dep = df_all_dep.loc[[row],:]
        nom_dep = df_dep['nom_dep'].iloc[0]
        ax_nom_dep = fig.add_subplot(grid[row+1, 0])
        make_header(ax_nom_dep, nom_dep, fontsize=14, halign='right')

        dep = df_dep['dep'].iloc[0]
        div_dose1 = grid[row+1, 1].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose1 = fig.add_subplot(div_dose1[0,0])
        make_bullet(ax_dose1, df_france, df_dep, dep, dose=1)

        div_dose2 = grid[row+1, 2].subgridspec(1, 2, width_ratios=[6,1])
        ax_dose2 = fig.add_subplot(div_dose2[0,0])
        make_bullet(ax_dose2, df_france, df_dep, dep, dose=2)

        df_inj = df_dep['mm_injections'].iloc[0]
        ax_spark =  fig.add_subplot(grid[row+1, 3])
        make_sparkline(ax_spark, df_inj)
        
        ax_card = fig.add_subplot(grid[row+1, 4])
        make_card(ax_card, df_inj)

        # plt.subplots_adjust(left=0, right=0.9)
        plt.xticks([])
        plt.yticks([])
    
    return header_fig, fig;
    


# %%

def check_dep(dep, age):
    global vacc, departements, result

    df1 = vacc[(vacc['dep'] == dep) & (vacc['clage_vacsi'] == age)].copy().set_index('jour')
    df1['total'] = df1.n_dose1 + df1.n_complet
    df1['mm'] = df1.total.rolling(7).mean()
    to_plot_mm = df1[['mm']].iloc[-30:]
    to_plot_inj = df1[['total']].iloc[-30:]

    df2 = result['all'].copy()
    df2 = df2[(df2['age'] == age) & (df2['dep'] == dep)].copy()
    df_inj = df2['mm_injections'].iloc[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 3))

    ax[0].plot(to_plot_mm)
    ax[1].plot(to_plot_inj)
    make_card(ax[2], df_inj)

    plt.show()
    return (to_plot_mm, to_plot_inj)

