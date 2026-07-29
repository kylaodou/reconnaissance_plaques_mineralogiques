"""
Interface Streamlit du projet de reconnaissance de plaques mineralogiques.

Reprend fidelement la structure de plaques_maquette.html : en-tete plein
cadre, zone de televersement, bouton, grille des etapes, carte de resultat.

Lancement : streamlit run app.py
"""

import os
import time
import cv2
import numpy as np
import streamlit as st

from anpr_pipeline import redimensionner, executer_pipeline_image


st.set_page_config(page_title="Reconnaissance de plaques burkinabè", layout="wide")

_ICI = os.path.dirname(__file__)
_DOSSIER_TELEVERSES = os.path.join(_ICI, "..", "images", "televerses")

with open(os.path.join(_ICI, "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def bgr_vers_rgb(img):
    if img is None:
        return None
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sauvegarder_televersement(fichier):
    """Conserve une copie de chaque photo televersee (utile pour ameliorer
    le pipeline sur des cas reels rencontres par les utilisateurs)."""
    try:
        os.makedirs(_DOSSIER_TELEVERSES, exist_ok=True)
        horodatage = time.strftime("%Y%m%d_%H%M%S")
        chemin = os.path.join(_DOSSIER_TELEVERSES, f"{horodatage}_{fichier.name}")
        with open(chemin, "wb") as f:
            f.write(fichier.getvalue())
    except Exception:
        pass  # la sauvegarde est un a-cote : elle ne doit jamais bloquer l'analyse


# ---------------------------------------------------------------------------
# En-tete (identique a la maquette : bandeau plein cadre, titre seul)
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="entete"><h1>Reconnaissance de plaques minéralogiques burkinabè</h1></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Zone de televersement + bouton (.bloc-image de la maquette)
# ---------------------------------------------------------------------------

fichier_televerse = st.file_uploader(
    "Choisir une image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

image_bgr, nom_source = None, None
if fichier_televerse is not None:
    _cle_fichier = (fichier_televerse.name, fichier_televerse.size)
    if st.session_state.get("dernier_fichier") != _cle_fichier:
        sauvegarder_televersement(fichier_televerse)
        st.session_state["dernier_fichier"] = _cle_fichier
    octets = np.frombuffer(fichier_televerse.getvalue(), np.uint8)
    image_bgr = cv2.imdecode(octets, cv2.IMREAD_COLOR)
    nom_source = fichier_televerse.name
    st.image(bgr_vers_rgb(image_bgr), caption="Aperçu", width=450)

_, col_bouton = st.columns([3, 2])
with col_bouton:
    lancer = st.button("Lancer la reconnaissance", type="primary",
                        disabled=image_bgr is None, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Execution + affichage (grille des etapes, puis carte de resultat)
# ---------------------------------------------------------------------------

if lancer and image_bgr is not None:
    with st.spinner("Traitement en cours..."):
        image_bgr = redimensionner(image_bgr)
        res = executer_pipeline_image(image_bgr, source=nom_source)

    etapes = [
        (res["image_originale"], "1. Image originale", False),
        (res["gris_lisse"], "2. Prétraitement", True),
        (res["contours_bruts"], "3. Contours (Canny)", True),
        (res["contours_fermes"], "4. Fermeture morphologique", True),
        (res["image_annotee"], "5. Localisation", False),
        (res["plaque_couleur"], "6. Plaque extraite (redressée)", False),
        (res["binaire"], "7. Binarisation", True),
        (res["binaire_annote"], "8. Segmentation", False),
    ]
    for ligne in (etapes[0:4], etapes[4:8]):
        cols = st.columns(4)
        for col, (img, titre, gris) in zip(cols, ligne):
            with col:
                with st.container(border=True):
                    st.image(bgr_vers_rgb(img), use_container_width=True, clamp=True)
                    st.markdown(f'<div class="legende-etape">{titre}</div>', unsafe_allow_html=True)

    detail = res.get("detail_reconnaissance") or []
    with st.container(border=True):
        st.markdown('<div class="legende-etape">9. Reconnaissance des caractères '
                    '(comparaison à la base de gabarits réels)</div>', unsafe_allow_html=True)
        if detail:
            colonnes = st.columns(len(detail))
            for colonne, d in zip(colonnes, detail):
                with colonne:
                    if d["vignette"] is not None:
                        st.image(cv2.bitwise_not(d["vignette"]), use_container_width=True)
                    st.markdown(
                        f"<div style='text-align:center;font-size:2em'>→ <b>{d['caractere']}</b></div>"
                        f"<div style='text-align:center;font-size:1.3em;color:#888'>{d['score']:.0%}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Aucun caractère segmenté à comparer.")

    st.markdown("<hr>", unsafe_allow_html=True)

    resultat_principal = res["texte_principal"]
    classe_vide = "" if resultat_principal else "vide"
    texte_affiche = resultat_principal if resultat_principal else "Aucun caractère reconnu"
    n_car = len(res["boites_caracteres"])
    statut = (
        f"{n_car} caractère(s) segmenté(s) sur {res['nb_lignes']} ligne(s) détectée(s) "
        f"&middot; plaque {'localisée' if res['plaque_trouvee'] else 'traitée en entier'}"
    )

    st.markdown(
        f"""
        <div class="carte-resultat">
            <div class="resultat-label">Résultat obtenu</div>
            <div class="plaque-lue {classe_vide}">{texte_affiche}</div>
            <div class="statut-discret">{statut}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
