import numpy as np
import cv2 as cv
import os  # Pour gérer le dossier de sauvegarde

# 1. Charger la vidéo
cap = cv.VideoCapture('Video/bugs2.mp4')

# --- CONFIGURATION DE L'ENREGISTREMENT ---
# A. Créer le dossier s'il n'existe pas
if not os.path.exists('Resultats'):
    os.makedirs('Resultats')

# B. Récupérer la taille et les FPS de la vidéo source
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# C. Initialiser l'enregistreur vidéo
# On sauvegarde sous le nom 'lucas_kanade_resultat.mp4'
out = cv.VideoWriter('Resultats/lucas_kanade_resultat.mp4', 
                     cv.VideoWriter_fourcc(*'mp4v'), 
                     fps, 
                     (frame_width, frame_height))
# -----------------------------------------

# Paramètres pour la détection des coins (GoodFeaturesToTrack)
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Paramètres pour le flux optique de Lucas-Kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Couleurs aléatoires pour dessiner les pistes
color = np.random.randint(0,255,(100,3))

# Prendre la première frame et trouver les coins
ret, old_frame = cap.read()
if not ret:
    print("Erreur de lecture de la vidéo")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Créer un masque pour dessiner les lignes (trajectoires)
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 2. Calculer le Flux Optique (Lucas-Kanade)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Sélectionner les bons points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # Dessiner les pistes
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # Convertir en entiers pour le dessin
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        
    img = cv.add(frame,mask)

    cv.imshow('Flux Optique Lucas-Kanade',img)

    # --- ENREGISTRER L'IMAGE DANS LA VIDÉO ---
    out.write(img)
    # -----------------------------------------

    # Mise à jour pour la prochaine itération
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    k = cv.waitKey(30) & 0xff
    if k == 27: # Touche ESC pour quitter
        break

cap.release()
out.release() # IMPORTANT : Fermer le fichier vidéo correctement
cv.destroyAllWindows()

print("Vidéo sauvegardée dans 'Resultats/lucas_kanade_resultat.mp4'")