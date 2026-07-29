# Reconnaissance de plaques minéralogiques burkinabè

Projet pratique du cours *Reconnaissance de motifs* (IBAM / Université Joseph KI-ZERBO), encadré par Jean Serge Dimitri Ouattara.

Chaîne complète de reconnaissance de plaques d'immatriculation burkinabè : prétraitement, localisation de la plaque (contours + croissance de région), binarisation, segmentation des caractères, et reconnaissance par comparaison de chaque caractère à une base de gabarits réels (mesure de similarité complémentaire CSM), sans OCR ni modèle appris.

## Contenu

- `code/anpr_pipeline.py` — la chaîne de traitement complète.
- `code/app.py` — interface web interactive (Streamlit).
- `images/plaques_bf/` — jeu de test documentaire (6 plaques burkinabè).
- `images/gabarits/` — base de gabarits de caractères réels utilisée pour la reconnaissance.
- `images/resultats/` — figures et résultats chiffrés (CSV) sur les deux jeux de test.
- `images/captures_app/` — captures d'écran de l'application.
- `Rapport_Reconnaissance_Plaques_Mineralogiques.pdf` — rapport complet (méthodologie, implémentation, résultats, discussion).

## Lancer l'application

```bash
cd code
pip install opencv-python numpy streamlit
streamlit run app.py
```
