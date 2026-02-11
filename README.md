# Estimation de Mouvement par Flux Optique (Optical Flow)

Ce projet porte sur la mise en œuvre et l'analyse de techniques de vision par ordinateur pour l'estimation du mouvement dans une séquence vidéo.

##  Objectif
L'objectif principal est de suivre des points d'intérêt caractéristiques (*features*) à travers une vidéo d'insectes (chenilles) pour visualiser leur champ de déplacement.

##  Méthodologies implémentées

Le projet explore deux approches complémentaires utilisant **OpenCV** et **Python** :

### 1. Méthode locale : Lucas-Kanade (Sparse Optical Flow)
Cette méthode suit un ensemble de points d'intérêt spécifiques (les "coins") d'une image à l'autre.
* **Détection :** Utilisation de `cv.goodFeaturesToTrack` (Shi-Tomasi) pour identifier les points d'ancrage fiables (fort gradient dans deux directions).
* **Suivi :** Utilisation de `cv.calcOpticalFlowPyrLK` avec gestion des pyramides d'images pour la robustesse.
* **Paramètres optimisés :** `qualityLevel = 0.15` et `maxLevel = 3`.

### 2. Méthode globale : Farneback (Dense Optical Flow)
Contrairement à Lucas-Kanade, cette méthode calcule le mouvement pour chaque pixel de l'image.
* **Visualisation :** Utilisation de l'espace couleur **HSV**.
    * **Teinte (Hue) :** Représente la direction du mouvement.
    * **Valeur (Value) :** L'intensité lumineuse représente la vitesse.
* **Avantage :** Permet une segmentation efficace par le mouvement et une vision d'ensemble des "masses" en déplacement.

##  Analyse des résultats

* **Sensibilité (qualityLevel) :** Un seuil trop bas ($0.1$) génère du bruit (sur-détection), tandis qu'un seuil trop haut ($0.3$) ignore les objets à faible contraste comme la chenille orange.
* **Robustesse (maxLevel) :** L'utilisation de pyramides (Niveau 3) est indispensable pour capturer les mouvements rapides et brusques des insectes.
* **Comparaison :** Lucas-Kanade est idéal pour le suivi précis de trajectoires (*tracking*), tandis que Farneback excelle dans la détection de l'activité globale et la segmentation de formes déformables.

##  Installation et Utilisation

### Prérequis
* Python 3.x
* OpenCV (`opencv-python`)
* NumPy

