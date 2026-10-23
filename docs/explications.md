# Fonctionnement de TWIZZY

TWIZZY reconnaît des panneaux de signalisation à partir d’une image, d’une vidéo ou d’une webcam. La chaîne part d’un filtrage couleur, isole les formes rondes, puis classe le panneau. Le classement final repose sur un réseau de neurones.

## Segmentation du rouge en HSV

Le rouge d’un panneau est cherché dans l’espace HSV (teinte, saturation, valeur), plus stable que le BGR quand la lumière change.

![Espace de couleur HSV](images/hsv-espace.jpeg)

Le rouge est aux deux bouts du cercle de teinte : orange vers 0° et violet vers 180°. Deux masques sont donc calculés, puis réunis par un OU. Un seuil de saturation écarte le blanc, et un seuil de valeur écarte le noir.

| Paramètre | Seuil bas | Seuil haut |
| --- | --- | --- |
| Teinte rouge-orange (H) | 0 | 6 |
| Teinte rouge-violet (H) | 170 | 180 |
| Saturation (S) | 110 | 255 |
| Valeur (V) | 35 | 255 |

## Cercles par ratio d’aire

Sur le masque, un filtre de Canny garde les contours. Pour chaque contour, on compare son aire à celle du cercle couvrant minimal :

Ratio = aire du contour / (π × r²) ≥ 0,8

Au-dessus de 0,8, la forme est gardée comme un cercle, même un peu déformée en ellipse.

![Panneaux ronds extraits d’une scène routière](images/extraction-cercles.png)

Cette méthode échoue si le contour n’est pas d’un seul tenant : un panneau coupé par un autre objet est vu comme plusieurs morceaux, et le cercle couvrant n’a plus de sens.

![Contours discontinus : le panneau partiellement masqué n’est plus un seul cercle](images/limites-cercle.png)

## Transformée de Hough

Hough retrouve un cercle même si le contour est interrompu, tant qu’il reste assez de pixels. Avant le vote, le masque est nettoyé : ouverture pour enlever le bruit, fermeture pour reboucher les trous, puis flou gaussien.

![Cercles retrouvés par Hough, y compris un panneau en partie caché](images/hough-resultat.png)

Hough a deux limites. De face seulement : un panneau en perspective devient une ellipse et les votes ne se rejoignent plus. Et il vise souvent l’intérieur de l’anneau rouge, ce qui laisse un résidu autour du disque extrait.

![Hough détecte l’intérieur de l’anneau du panneau 110](images/hough-interieur.png)

Les deux méthodes sont donc enchaînées :

1. Le ratio d’aire extrait d’abord les panneaux nets.
2. Chaque panneau trouvé est effacé du masque par un disque noir plein.
3. Hough ne voit plus que les formes restantes, les panneaux fragmentés.

## Reconnaissance par matching ORB

Les disques extraits étaient d’abord comparés à des panneaux de référence avec ORB : points d’intérêt, descripteurs, puis score de ressemblance. La référence au score le plus haut donne la classe.

![Correspondance ORB du panneau 50 avec les références](images/matching-50.png)

ORB confond facilement 50 et 90, et rate le panneau dès qu’il n’est plus de face.

![La détection alterne entre 50 et 90 km/h](images/detection-90.png)

C’est pour ça que la reconnaissance finale passe par un réseau de neurones.

## DeepLearning4J

Un premier CNN, en Java avec DeepLearning4J, a servi de test. Base de 11 621 images et 87 classes, séparée en 80 % d’entraînement et 20 % de test.

| Couche | Type | Filtres ou neurones | Activation |
| --- | --- | --- | --- |
| 1 | Convolution + max pooling | 16 filtres 3×3 | ReLU |
| 2 | Convolution + max pooling | 32 filtres 3×3 | ReLU |
| 3 | Couche dense | 128 neurones | ReLU |
| 4 | Sortie | 87 classes | Softmax |

Le modèle reconnaît la plupart des panneaux, mais se trompe encore sur certains, surtout quand une classe a peu d’exemples.

![Stationnement interdit](images/dl4j-1.png)
![Arrêt et stationnement interdits](images/dl4j-2.png)
![Sens unique, classe mal reconnue par DeepLearning4J](images/dl4j-3.png)
![Virage dangereux à gauche](images/dl4j-4.png)

## CNN Keras, puis ONNX

Le modèle utilisé ensuite est entraîné en Python avec Keras, sur 37 948 images et 97 classes. La base Roboflow contient souvent plusieurs panneaux par photo : YOLOv8 les localise (confiance ≥ 0,44), chaque boîte est recadrée, puis classée. Les classes sont rééquilibrées à au moins 300 images, avec rotation, miroir, luminosité, zoom et décalage.

```text
Conv2D 32, 3×3, ReLU, entrée 64×64×3
MaxPooling 2×2
Conv2D 64, 3×3, ReLU
MaxPooling 2×2
Conv2D 128, 3×3, ReLU
MaxPooling 2×2
Flatten
Dense 256, ReLU
Dropout 0,5
Dense 97, softmax
```

L’entraînement se stabilise en moins de 30 époques, autour de 97 % de précision. Les courbes d’entraînement et de validation restent proches : le modèle généralise, sans surapprentissage visible.

![Précision et perte du CNN Keras](images/courbes-keras.png)

En utilisation, YOLOv8 garde les détections au-dessus de 0,35 et le CNN celles au-dessus de 0,70. La caméra visée est en 1280×720 à 30 images par seconde.

![Reconnaissance sur un flux vidéo](images/keras-video.png)
![Reconnaissance sur une image fixe](images/keras-image.png)

Le modèle Keras est converti en ONNX (opset 13) pour tourner dans Java sans TensorFlow. L’entrée est un lot d’images 64×64×3, la sortie est un vecteur de 97 classes. Sur les mêmes panneaux, le sens unique mal classé par DeepLearning4J est cette fois correct.

![Limitation 120](images/onnx-1.png)
![Limitation 30](images/onnx-2.png)
![Sens unique, correctement reconnu](images/onnx-3.png)
![Stationnement interdit](images/onnx-4.png)

## Interface

Au lancement, `Main` ouvre `HomeScreen`, puis `ChoiceCNNScreen` pour choisir le modèle : Python, DeepLearning4J ou ONNX. `ChoiceSRCScreen` choisit ensuite l’image, la vidéo ou la webcam.

![Enchaînement des écrans et des moteurs de reconnaissance](images/uml-interface.png)

![Choix du modèle](images/interface-modele.png)
![Choix de la source](images/interface-source.png)
![Sélection du fichier](images/interface-fichier.png)
![Détection affichée sur l’image](images/interface-detection.png)

## Architecture prévue au fil du projet

Deux découpages ont été dessinés avant l’interface finale. Le premier sépare la lecture vidéo, le traitement d’image et une base de descripteurs.

![Lecture vidéo, traitement d’image et base de descripteurs](images/architecture-generale.png)

Le second détaille les paquets : prétraitement HSV, contours, descripteurs ORB, modèle de données, vidéo et affichage.

![Découpage complet envisagé pour le pipeline](images/architecture-complete.png)

L’application livrée suit plutôt le diagramme de l’interface : trois moteurs de reconnaissance derrière les mêmes écrans.

## Ce qui reste difficile

Le matching ORB n’est pas fiable dès que la base de référence grandit ou que le panneau est de biais. DeepLearning4J est plus lourd à entraîner que Keras, d’où sa base plus petite et ses erreurs. Lancer le modèle Python depuis la fenêtre Java reste bloqué par les bibliothèques : Java ne trouve pas l’environnement Keras. La suite prévue est une meilleure tenue de nuit et par mauvais temps, d’autres familles de panneaux, un réseau plus léger, puis un essai sur le véhicule.
