import streamlit as st

def display_desc():

    st.header("Description et interprétation des indicateurs")

    st.markdown(
        '''
        **Pourcentage de la population partiellement vaccinée** : pourcentage de la population sélectionnée ayant reçu une dose de vaccin pour les vaccins nécessitant deux doses (Pfizer-BioNTech, Moderna, AstraZeneca).
        Certaines personnes peuvent avoir besoin de trois doses pour être totalement vaccinée.
        La petite barre verticale noire indique le niveau pour la France entière

        **Pourcentage de la population entièrement vaccinée** : pourcentage de la population sélectionnée ayant reçu toutes les doses requises, c'est-à-dire deux doses pour les vaccins Pfizer-BioNTech, Moderna, AstraZeneca, une dose pour le vaccin Janssen, trois doses pour les personnes concernées.
        La petite barre verticale noire indique le niveau pour la France entière
        
        **Pourcentage des primo-vaccinations sur le total des injections, 7 jours glissants** : le rapport du nombre de personnes ayant reçu une première dose de vaccins (pour les vaccins nécessitant deux doses) dans les 7 derniers jours sur le nombre total de personnes ayant reçu une injection sur la même période.

        Cet indicateur permet d'évaluer la dynamique de la campagne de vaccination. Un pourcentage élevé indique un recrutement important de nouveaux candidats à la vaccination parmi la population. 
        Attention cependant, car ce chiffre diminue naturellement au fur et à mesure que le taux de couverture dans la classe d'âge progresse. D'autre part, il diminue quand le nombre de personnes qui n'ont besoin que d'une seule dose augmente. 

        **Nombre d'injections pour 100 personnes non vaccinées, 7 jours glissants** : rapport du nombre total d'injections effectuées dans les 7 derniers jours sur le nombre d'injections qu'il restait à effectuer au début de la période pour couvrir toute la population.

        On suppose qu'il reste à faire deux injections pour chaque personne non vaccinée et une injection pour les personnes partiellement vaccinées. Cet indicateur permet d'évaluer le délai au bout duquel toute la population sera protégée. Par exemple, une valeur de 5 indique que si les vaccinations se poursuivent au même rythme, il faudra encore environ 20 semaines pour vacciner toute la population. 
        Cette estimation est biaisée par la proportion de personnes qui n'ont besoin que d'une injection. 

        **Incidence parmi les personnes non vaccinées, moyenne glissante 7 jours** : l'incidence est le nombre de nouveaux cas détectés pour 100 000 habitants. Ici, au lieu de diviser par l'ensemble de la population, on divise par le nombre de personnes qui ne sont pas encore entièrement vaccinées. 

        La protection conférée par les vaccins étant très bonne, il est raisonnable de supposer que la majorité des cas de Covid concernent des personnes qui ne sont pas encore vaccinées. Cet indicateur permet donc d'évaluer la circulation du virus dans cette population qui pourrait être masquée par l'effet de la vaccination si on considérait toute la population. 
        '''
    )
def display_att(): 
    st.header("Points d'attention")
    st.markdown(
        '''
        - Les classes d'âge sont découpées différemment entre les données de vaccination et les données d'incidence. Les pourcentages de couverture vaccinale ont donc dus être recalculés sur des classes d'âge différentes ce qui entraîne des imprécisions, en particulier pour la classe d'âge des 18-29 ans pour laquelle une extrapolation a été nécessaire.

        - Malgré toute l'attention que j'ai portée à l'élaboration de ce tableau, des erreurs peuvent être présentes. Tous les indicateurs doivent donc être pris avec une petite dose de scepticisme. Si vous constatez des erreurs ou pour toute autre remarque vous pouvez me contacter sur Twitter [@FranklinMaillot](https://twitter.com/FranklinMaillot). 

        - Plus de détails sur les indicateurs et leur interprétation sous le tableau. 

        '''
    )

