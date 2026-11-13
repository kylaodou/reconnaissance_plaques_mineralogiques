# Reconnaissance de plaques minéralogiques burkinabè

Projet pratique du cours *Reconnaissance de motifs* (IBAM / Université Joseph KI-ZERBO), encadré par Jean Serge Dimitri Ouattara.

Chaîne complète de reconnaissance de plaques d'immatriculation burkinabè : prétraitement, localisation de la plaque (contours + croissance de région), binarisation, segmentation des caractères, et reconnaissance par comparaison de chaque caractère à une base de gabarits réels (mesure de similarité complémentaire CSM), sans OCR ni modèle appris.

## Étapes de la chaîne

1. **Prétraitement** : redimensionnement, passage en niveaux de gris, lissage.
2. **Détection des contours** et recherche des zones candidates : par la forme (contours fermés
   au bon rapport largeur / hauteur) et par la couleur du fond de plaque.
3. **Choix de la plaque** parmi les meilleurs candidats, puis redressement de la perspective.
4. **Binarisation** : plusieurs variantes sont essayées et la meilleure est retenue selon la
   qualité de la segmentation obtenue.
5. **Segmentation des caractères** : composantes connexes, découpage par projection pour séparer
   les caractères collés, filtrage des formes incohérentes, regroupement en lignes.
6. **Reconnaissance** : chaque caractère est comparé aux gabarits de `images/gabarits/` avec la
   mesure CSM ; un seuil de confiance écarte les correspondances douteuses.

## Contenu

- `code/anpr_pipeline.py` — la chaîne de traitement complète.
- `code/app.py` — interface web interactive (Streamlit).
- `code/style.css` — mise en forme de l'interface.
- `images/gabarits/` — base de gabarits de caractères réels utilisée pour la reconnaissance.

## Prérequis

- Python 3.10 ou plus
- pip

Dépendances : `streamlit`, `opencv-python-headless`, `numpy`.

## Lancer l'application

```bash
git clone https://github.com/kylaodou/reconnaissance_plaques_mineralogiques.git
cd reconnaissance_plaques_mineralogiques/code
pip install -r requirements.txt
streamlit run app.py
```

L'interface s'ouvre sur http://localhost:8501. Téléversez une photo de véhicule : l'application
affiche chaque étape (plaque détectée, binarisation, caractères segmentés) et le texte reconnu.

## Utiliser la chaîne sans interface

```python
from anpr_pipeline import executer_pipeline

resultat = executer_pipeline("chemin/vers/photo.jpg")
```

## Déploiement

Le fichier `requirements.txt` à la racine et le dossier `.streamlit/` permettent de déployer
l'application sur Streamlit Community Cloud en indiquant `code/app.py` comme fichier principal.
