# Reconnaissance de plaques minéralogiques burkinabè

Projet pratique du cours *Reconnaissance de motifs* (IBAM / Université Joseph KI-ZERBO), encadré par Jean Serge Dimitri Ouattara.

Chaîne complète de reconnaissance de plaques d'immatriculation burkinabè : prétraitement, localisation de la plaque (contours + croissance de région), binarisation, segmentation des caractères, et reconnaissance par comparaison de chaque caractère à une base de gabarits réels (mesure de similarité complémentaire CSM), sans OCR ni modèle appris.

## Contenu

- `code/anpr_pipeline.py` — la chaîne de traitement complète.
- `code/app.py` — interface web interactive (Streamlit).
- `images/gabarits/` — base de gabarits de caractères réels utilisée pour la reconnaissance.

## Lancer l'application

```bash
cd code
pip install -r requirements.txt
streamlit run app.py
```
