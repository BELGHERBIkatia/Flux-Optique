import numpy as np
import cv2 as cv
import os  # NOUVEAU : Pour gérer les dossiers

# 1. Charger la vidéo
video_path = 'Video/bugs2.mp4'
cap = cv.VideoCapture(video_path)

# Lecture de la première frame
ret, frame1 = cap.read()
if not ret:
    print("Erreur de lecture de la vidéo")
    exit()

# --- NOUVEAU : CONFIGURATION DE L'ENREGISTREMENT ---
# A. Créer le dossier s'il n'existe pas
output_folder = 'Resultats'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# B. Récupérer les infos de la vidéo source (largeur, hauteur, fps)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# C. Définir le chemin de sortie et le codec (mp4v pour .mp4)
output_path = os.path.join(output_folder, 'flux_dense_resultat.mp4')
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"L'enregistrement a commencé dans : {output_path}")
# ----------------------------------------------------

# Conversion en niveaux de gris
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Création de l'image HSV
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # 2. Calcul du Flux Optique Dense
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 
                                       pyr_scale=0.5, 
                                       levels=3, 
                                       winsize=15, 
                                       iterations=3, 
                                       poly_n=5, 
                                       poly_sigma=1.2, 
                                       flags=0)

    # 3. Visualisation HSV
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Afficher le résultat
    cv.imshow('Flux Dense (Farneback) - HSV', bgr)
    
    # --- NOUVEAU : Ecrire l'image dans le fichier vidéo ---
    out.write(bgr)
    # ------------------------------------------------------

    # Mise à jour pour la prochaine itération
    prvs = next

    k = cv.waitKey(30) & 0xff
    if k == 27: # Touche ESC pour quitter
        break

# Libération des ressources
cap.release()
out.release() # NOUVEAU : Important pour finaliser le fichier vidéo
cv.destroyAllWindows()

print("Enregistrement terminé.")