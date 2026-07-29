import os
import cv2
import numpy as np

ALPHABET_PLAQUE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TAILLE_MAX_COTE = 1200  # redimensionnement des images d'entree


# ---------------------------------------------------------------------------
# Etape 0 : acquisition
# ---------------------------------------------------------------------------

def redimensionner(image, taille_max=TAILLE_MAX_COTE):
    h, w = image.shape[:2]
    echelle = taille_max / max(h, w)
    if echelle < 1.0:
        image = cv2.resize(image, (int(w * echelle), int(h * echelle)), interpolation=cv2.INTER_AREA)
    return image


def charger_et_redimensionner(chemin, taille_max=TAILLE_MAX_COTE):
    image = cv2.imread(chemin)
    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {chemin}")
    return redimensionner(image, taille_max)


# ---------------------------------------------------------------------------
# Etape 1 : pretraitement
# ---------------------------------------------------------------------------

def pretraiter(image_bgr):
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gris_lisse = cv2.bilateralFilter(gris, d=9, sigmaColor=75, sigmaSpace=75)
    return gris, gris_lisse


# ---------------------------------------------------------------------------
# Etape 2 : segmentation par contours -> localisation de la plaque
# ---------------------------------------------------------------------------

def detecter_contours(gris_lisse):
    contours_bruts = cv2.Canny(gris_lisse, 50, 150)
    noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    contours_fermes = cv2.morphologyEx(contours_bruts, cv2.MORPH_CLOSE, noyau)
    return contours_bruts, contours_fermes


def candidats_par_contour(contours_fermes, aire_image):
    contours, _ = cv2.findContours(contours_fermes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidats = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aire_rect = w * h
        if aire_rect < 0.004 * aire_image or aire_rect > 0.60 * aire_image:
            continue
        ratio = w / float(h)
        if not (1.3 <= ratio <= 6.5):
            continue
        remplissage = cv2.contourArea(c) / float(aire_rect)
        candidats.append((remplissage * aire_rect, (x, y, w, h), c))
    return candidats


def candidats_par_couleur(image_bgr, aire_image, pas_grille=40, tolerance=(12, 50, 50)):
    """Segmentation par region (approche ascendante) : des germes repartis
    sur une grille font croitre la region connexe de teinte proche du
    germe. Complementaire aux contours, utile quand la plaque a un fond de
    couleur unie (jaune, bleu) mal detecte par le gradient."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = image_bgr.shape[:2]
    visite = np.zeros((h_img + 2, w_img + 2), dtype=np.uint8)

    candidats = []
    for gy in range(pas_grille // 2, h_img, pas_grille):
        for gx in range(pas_grille // 2, w_img, pas_grille):
            if visite[gy + 1, gx + 1]:
                continue
            masque = np.zeros((h_img + 2, w_img + 2), dtype=np.uint8)
            cv2.floodFill(hsv, masque, (gx, gy), 0, loDiff=tolerance, upDiff=tolerance,
                          flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (1 << 8))
            visite |= masque

            region = masque[1:-1, 1:-1]
            aire = int(region.sum())
            if aire < 0.004 * aire_image or aire > 0.5 * aire_image:
                continue
            ys, xs = np.nonzero(region)
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
            ratio = w / float(h)
            if not (1.3 <= ratio <= 6.5):
                continue
            remplissage = aire / float(w * h)
            if remplissage < 0.5:
                continue
            contours, _ = cv2.findContours(region[y:y + h, x:x + w],
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) + (x, y) if contours else None
            candidats.append((remplissage * w * h, (x, y, w, h), contour))
    return candidats


def rogner_avec_marge(image, boite, marge_ratio=0.06):
    x, y, w, h = boite
    mx, my = int(w * marge_ratio), int(h * marge_ratio)
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w_img, x + w + mx), min(h_img, y + h + my)
    return image[y0:y1, x0:x1]


def _ordonner_points(pts):
    """Ordonne 4 points en [haut-gauche, haut-droite, bas-droite, bas-gauche]."""
    pts = pts[np.argsort(pts[:, 1])]
    haut = pts[:2][np.argsort(pts[:2, 0])]
    bas = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([haut[0], haut[1], bas[1], bas[0]], dtype=np.float32)


def extraire_plaque_redressee(image_bgr, boite, contour=None, marge_ratio=0.03):
    """Corrige la perspective si la plaque est photographiee de biais
    (transformation a partir des 4 coins du contour) ; sinon recadrage simple."""
    if contour is None or len(contour) < 5:
        return rogner_avec_marge(image_bgr, boite, marge_ratio)

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < 4 or rh < 4:
        return rogner_avec_marge(image_bgr, boite, marge_ratio)

    rw_m, rh_m = rw * (1 + 2 * marge_ratio), rh * (1 + 2 * marge_ratio)
    src = _ordonner_points(cv2.boxPoints(((cx, cy), (rw_m, rh_m), angle)))

    largeur = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))
    hauteur = max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))
    if largeur < 8 or hauteur < 8:
        return rogner_avec_marge(image_bgr, boite, marge_ratio)

    if hauteur > largeur:  # une plaque est toujours plus large que haute
        src = np.array([src[1], src[2], src[3], src[0]], dtype=np.float32)
        largeur, hauteur = hauteur, largeur

    # correction appliquee seulement si l'angle mesure est plausible
    vecteur_haut = src[1] - src[0]
    angle_mesure = abs(np.degrees(np.arctan2(vecteur_haut[1], vecteur_haut[0])))
    if angle_mesure < 1.5 or angle_mesure > 20.0:
        return rogner_avec_marge(image_bgr, boite, marge_ratio)

    dst = np.array([[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (int(largeur), int(hauteur)))


# ---------------------------------------------------------------------------
# Etape 3 : segmentation par regions -> binarisation de la plaque
# ---------------------------------------------------------------------------

def variantes_binarisation(plaque_gris):
    """Seuillage d'Otsu et seuillage adaptatif, chacun dans les deux
    polarites possibles : plusieurs hypotheses de binarisation, departagees
    plus loin par la qualite de la segmentation en caracteres qu'elles
    produisent (segmenter_plaque)."""
    variantes = []

    _, otsu = cv2.threshold(plaque_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(("otsu", cv2.bitwise_not(otsu)))
    variantes.append(("otsu_inverse", otsu))

    bloc = max(15, (min(plaque_gris.shape) // 6) | 1)
    adapt = cv2.adaptiveThreshold(plaque_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, bloc, 9)
    variantes.append(("adaptatif", cv2.bitwise_not(adapt)))
    variantes.append(("adaptatif_inverse", adapt))

    return variantes


# ---------------------------------------------------------------------------
# Etape 4 : segmentation des caracteres (composantes connexes)
# ---------------------------------------------------------------------------

def _composantes_brutes(binaire, tolerance_bord=2):
    """Ecarte les composantes trop grandes touchant le bord (cadre de la
    plaque) et celles de hauteur incompatible avec un caractere."""
    h_p, w_p = binaire.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaire, connectivity=8)

    boites = []
    for i in range(1, n):  # l'etiquette 0 est le fond
        x, y, w, h, aire = stats[i]
        touche_bord = (x <= tolerance_bord or y <= tolerance_bord
                      or x + w >= w_p - tolerance_bord or y + h >= h_p - tolerance_bord)
        trop_grande = (w > 0.5 * w_p or h > 0.9 * h_p)
        if touche_bord and trop_grande:
            continue
        if not (0.10 * h_p <= h <= 0.95 * h_p):
            continue
        if w > 0.65 * w_p:
            continue
        boites.append((int(x), int(y), int(w), int(h)))
    return boites


def _filtrer_forme_caractere(boites):
    """Un caractere est plus haut que large : ecarte le bruit ponctuel et
    les elements decoratifs trop plats ou trop filiformes."""
    return [(x, y, w, h) for (x, y, w, h) in boites if 0.06 <= w / float(h) <= 1.5]


def _boite_serree(binaire, x0, x1, y, h):
    region = binaire[y:y + h, x0:x1]
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return None
    return (int(x0 + xs.min()), int(y + ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _plages_actives(profil, seuil, largeur_min):
    plein = profil > seuil
    plages, debut = [], None
    for i, est_plein in enumerate(plein):
        if est_plein and debut is None:
            debut = i
        elif not est_plein and debut is not None:
            plages.append((debut, i))
            debut = None
    if debut is not None:
        plages.append((debut, len(plein)))
    return [p for p in plages if p[1] - p[0] >= largeur_min]


def _decouper_par_projection(binaire, boite, fraction_bande=None):
    """Decoupe une composante via son profil de projection verticale
    (vallee de colonnes vides entre deux symboles accoles). Avec
    fraction_bande, ne projette que la bande centrale de la boite~: un
    caractere la traverse toujours, contrairement a un element decoratif
    empile au-dessus (embleme, sigle)."""
    x, y, w, h = boite
    region = binaire[y:y + h, x:x + w]

    if fraction_bande is None:
        bande = region
        seuil = max(1, int(0.06 * h))
    else:
        r0 = int(h * (0.5 - fraction_bande / 2))
        r1 = max(r0 + 1, int(h * (0.5 + fraction_bande / 2)))
        bande = region[r0:r1, :]
        seuil = max(1, int(0.10 * (r1 - r0)))

    plages = _plages_actives((bande > 0).sum(axis=0), seuil, largeur_min=max(2, int(0.05 * w)))
    morceaux = [_boite_serree(binaire, x + a, x + b, y, h) for (a, b) in plages]
    return [m for m in morceaux if m is not None]


def _separer_composantes_larges(binaire, boites, facteur=1.6):
    """Detecte les composantes anormalement larges (embleme + sigle +
    premier chiffre souvent relies par la binarisation) et tente de les
    decouper~: d'abord sur la bande centrale, puis sur la hauteur entiere."""
    if len(boites) < 2:
        return boites
    largeurs = sorted(b[2] for b in boites)
    largeur_reference = largeurs[len(largeurs) // 2]

    resultat = []
    for boite in boites:
        if boite[2] <= facteur * largeur_reference:
            resultat.append(boite)
            continue
        candidats = (_decouper_par_projection(binaire, boite, fraction_bande=0.20)
                     or _decouper_par_projection(binaire, boite))
        utile = len(candidats) >= 2 or (len(candidats) == 1 and candidats[0][2] <= 0.8 * boite[2])
        resultat.extend(candidats if utile else [boite])
    return resultat


def _retenir_caracteres_coherents(boites):
    """Ne garde que les composantes dont la hauteur et la position
    verticale sont coherentes avec le reste du groupe (ecarte les
    artefacts, ex. l'embleme national, dont la hauteur peut par hasard
    etre plausible mais qui n'est pas assis sur la meme ligne de base)."""
    if len(boites) < 3:
        return boites
    hauteurs = sorted(b[3] for b in boites)
    reference_h = hauteurs[len(hauteurs) // 2]

    tops = sorted(b[1] for b in boites)
    reference_top = tops[len(tops) // 2]
    ecarts_top = sorted(abs(b[1] - reference_top) for b in boites)
    mad_top = ecarts_top[len(ecarts_top) // 2]
    tolerance_top = max(0.15 * reference_h, 3 * 1.4826 * mad_top)

    retenues = [
        b for b in boites
        if 0.6 * reference_h <= b[3] <= 1.6 * reference_h
        and abs(b[1] - reference_top) <= tolerance_top
    ]
    return retenues if len(retenues) >= 3 else boites


def grouper_en_lignes(boites, tolerance=0.6):
    """Regroupe les caracteres par ligne (centres verticaux proches), puis
    les ordonne de gauche a droite dans chaque ligne."""
    if not boites:
        return []
    boites = sorted(boites, key=lambda b: b[1] + b[3] / 2)
    lignes, courante = [], [boites[0]]
    for b in boites[1:]:
        centre_prec = courante[-1][1] + courante[-1][3] / 2
        centre_actuel = b[1] + b[3] / 2
        if abs(centre_actuel - centre_prec) > tolerance * b[3]:
            lignes.append(courante)
            courante = [b]
        else:
            courante.append(b)
    lignes.append(courante)
    return [sorted(ligne, key=lambda b: b[0]) for ligne in lignes]


def _score_segmentation(lignes):
    """Score de plausibilite d'une segmentation~: homogeneite de hauteur
    des caracteres (mediane robuste) et nombre de caracteres plausible
    pour une plaque (5 a 10). Sert a choisir entre les binarisations et
    les localisations candidates."""
    caracteres = [b for ligne in lignes for b in ligne]
    n = len(caracteres)
    if n < 4:
        return 0.0

    hauteurs = np.array([b[3] for b in caracteres], dtype=float)
    mediane = np.median(hauteurs)
    if mediane < 15:  # trop petit pour etre un caractere (bruit, texture)
        return 0.0
    mad = np.median(np.abs(hauteurs - mediane))
    homogeneite_h = 1.0 - min(1.0, (1.4826 * mad) / max(1.0, mediane))

    if 5 <= n <= 10:
        plausibilite = 1.0
    else:
        ecart = (5 - n) if n < 5 else (n - 10)
        plausibilite = 0.6 ** min(4, ecart)

    score = homogeneite_h * min(n, 10) / 10.0 * plausibilite
    if len(lignes) > 2:
        score *= 0.4
    return score


def segmenter_plaque(plaque_gris):
    """Essaie chaque binarisation candidate et garde celle qui donne la
    segmentation en caracteres la plus plausible."""
    meilleure = None
    for nom, binaire in variantes_binarisation(plaque_gris):
        boites = _composantes_brutes(binaire)
        boites = _separer_composantes_larges(binaire, boites)
        boites = _filtrer_forme_caractere(boites)
        boites = _retenir_caracteres_coherents(boites)
        lignes = grouper_en_lignes(boites)
        score = _score_segmentation(lignes)
        if meilleure is None or score > meilleure[0]:
            meilleure = (score, nom, binaire, lignes)

    score, nom, binaire, lignes = meilleure
    boites_ordonnees = [b for ligne in lignes for b in ligne]
    return binaire, boites_ordonnees, len(lignes), nom


def candidats_localisation(image_bgr, contours_fermes, top_k=12):
    """Rassemble les candidats de localisation par contour et par couleur."""
    aire_image = image_bgr.shape[0] * image_bgr.shape[1]
    candidats = candidats_par_contour(contours_fermes, aire_image)
    candidats += candidats_par_couleur(image_bgr, aire_image)
    candidats.sort(key=lambda t: t[0], reverse=True)
    return candidats[:top_k]


def choisir_localisation(image_bgr, candidats):
    """Evalue chaque candidat en executant sur lui extraction + redressement
    + segmentation complete, et retient celui dont la segmentation obtient
    le meilleur score (_score_segmentation) : plutot que de figer la
    localisation sur la seule force du contour, ce sont les caracteres
    effectivement segmentables qui departagent les candidats."""
    h_img, w_img = image_bgr.shape[:2]
    meilleur = None  # (score, boite, plaque_couleur, binaire, boites, nb_lignes, methode)

    for _score_brut, boite, contour in candidats:
        plaque_couleur = extraire_plaque_redressee(image_bgr, boite, contour)
        if plaque_couleur.size == 0:
            continue
        plaque_gris = cv2.cvtColor(plaque_couleur, cv2.COLOR_BGR2GRAY)
        binaire, boites, nb_lignes, methode = segmenter_plaque(plaque_gris)
        score = _score_segmentation(grouper_en_lignes(boites))
        if meilleur is None or score > meilleur[0]:
            meilleur = (score, boite, plaque_couleur, binaire, boites, nb_lignes, methode)

    if meilleur is not None and meilleur[0] > 0:
        _, boite, plaque_couleur, binaire, boites, nb_lignes, methode = meilleur
        return boite, True, plaque_couleur, binaire, boites, nb_lignes, methode

    # Repli : aucun candidat plausible -> l'image entiere.
    boite = (0, 0, w_img, h_img)
    plaque_couleur = rogner_avec_marge(image_bgr, boite, marge_ratio=0.0)
    plaque_gris = cv2.cvtColor(plaque_couleur, cv2.COLOR_BGR2GRAY)
    binaire, boites, nb_lignes, methode = segmenter_plaque(plaque_gris)
    return boite, False, plaque_couleur, binaire, boites, nb_lignes, methode


# ---------------------------------------------------------------------------
# Etape 5 : reconnaissance des caracteres (base de gabarits reels)
# ---------------------------------------------------------------------------

_TAILLE_GABARIT = (40, 60)  # (largeur, hauteur)
_DOSSIER_GABARITS = os.path.join(os.path.dirname(__file__), "..", "images", "gabarits")
_gabarits_caracteres = None


def _charger_gabarits_caracteres():
    """Charge la base de gabarits reels (un ou plusieurs echantillons
    photographies par caractere, voir images/gabarits/<caractere>/)."""
    gabarits = {}
    if not os.path.isdir(_DOSSIER_GABARITS):
        return gabarits
    for nom_dossier in os.listdir(_DOSSIER_GABARITS):
        if nom_dossier not in ALPHABET_PLAQUE:
            continue
        chemin_dossier = os.path.join(_DOSSIER_GABARITS, nom_dossier)
        images = []
        for nom_fichier in os.listdir(chemin_dossier):
            img = cv2.imread(os.path.join(chemin_dossier, nom_fichier), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, _TAILLE_GABARIT, interpolation=cv2.INTER_AREA))
        if images:
            gabarits[nom_dossier] = images
    return gabarits


def _similarite_csm(vignette, gabarit):
    """Mesure de similarite complementaire (CSM) entre deux images binaires
    de meme taille : a = pixels noirs communs, b = noir chez le gabarit
    seul, c = noir chez la vignette seule, e = blancs communs, n = pixels
    totaux, T = pixels noirs du gabarit. Renvoie un rapport dans [-1, 1]."""
    f = vignette > 127
    t = gabarit > 127
    a = int(np.count_nonzero(f & t))
    b = int(np.count_nonzero(~f & t))
    c = int(np.count_nonzero(f & ~t))
    e = int(np.count_nonzero(~f & ~t))
    n = f.size
    T = a + b
    denominateur = T * (n - T)
    return (a * e - b * c) / denominateur if denominateur else 0.0


def detailer_reconnaissance(binaire, boites, marge=4, seuil_confiance=0.2):
    """Pour chaque caractere segmente, compare sa vignette a tous les
    gabarits reels (CSM) et renvoie {vignette, caractere, score} -- le
    meilleur score par caractere, "?" si aucun ne depasse seuil_confiance."""
    global _gabarits_caracteres
    if _gabarits_caracteres is None:
        _gabarits_caracteres = _charger_gabarits_caracteres()

    details = []
    for (x, y, w, h) in boites:
        x0, y0 = max(0, x - marge), max(0, y - marge)
        x1 = min(binaire.shape[1], x + w + marge)
        y1 = min(binaire.shape[0], y + h + marge)
        vignette_brute = binaire[y0:y1, x0:x1]
        if vignette_brute.size == 0:
            details.append({"vignette": None, "caractere": "?", "score": 0.0})
            continue
        vignette = cv2.resize(vignette_brute, _TAILLE_GABARIT, interpolation=cv2.INTER_AREA)
        meilleur_c, meilleur_score = "?", seuil_confiance
        for c, echantillons in _gabarits_caracteres.items():
            for gabarit in echantillons:
                score = _similarite_csm(vignette, gabarit)
                if score > meilleur_score:
                    meilleur_score, meilleur_c = score, c
        details.append({"vignette": vignette_brute, "caractere": meilleur_c, "score": meilleur_score})
    return details


def reconnaitre_caracteres(binaire, boites, marge=4, seuil_confiance=0.2):
    """Reconnait chaque caractere segmente par comparaison CSM a la base de
    gabarits reels ; renvoie le texte complet dans l'ordre de lecture."""
    if not boites:
        return ""
    return "".join(d["caractere"] for d in detailer_reconnaissance(binaire, boites, marge, seuil_confiance))


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def executer_pipeline_image(image_bgr, source="image en memoire"):
    gris, gris_lisse = pretraiter(image_bgr)
    contours_bruts, contours_fermes = detecter_contours(gris_lisse)
    candidats = candidats_localisation(image_bgr, contours_fermes)

    (boite, plaque_trouvee, plaque_couleur, binaire,
     boites_caracteres, nb_lignes, methode) = choisir_localisation(image_bgr, candidats)

    image_annotee = image_bgr.copy()
    x, y, w, h = boite
    cv2.rectangle(image_annotee, (x, y), (x + w, y + h), (0, 0, 255), 3)

    plaque_gris = cv2.cvtColor(plaque_couleur, cv2.COLOR_BGR2GRAY)

    binaire_annote = cv2.cvtColor(binaire, cv2.COLOR_GRAY2BGR)
    for (bx, by, bw, bh) in boites_caracteres:
        cv2.rectangle(binaire_annote, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    resultat = {
        "chemin": source,
        "image_originale": image_bgr,
        "gris": gris,
        "gris_lisse": gris_lisse,
        "contours_bruts": contours_bruts,
        "contours_fermes": contours_fermes,
        "image_annotee": image_annotee,
        "plaque_trouvee": plaque_trouvee,
        "plaque_couleur": plaque_couleur,
        "plaque_gris": plaque_gris,
        "binaire": binaire,
        "binaire_annote": binaire_annote,
        "boites_caracteres": boites_caracteres,
        "nb_lignes": nb_lignes,
        "deux_lignes": nb_lignes >= 2,
        "methode_binarisation": methode,
    }
    detail = detailer_reconnaissance(binaire, boites_caracteres)
    resultat["detail_reconnaissance"] = detail
    resultat["texte_principal"] = "".join(d["caractere"] for d in detail)
    return resultat


def executer_pipeline(chemin_image):
    """Variante de executer_pipeline_image() qui part d'un chemin de fichier."""
    image_bgr = charger_et_redimensionner(chemin_image)
    return executer_pipeline_image(image_bgr, source=chemin_image)
