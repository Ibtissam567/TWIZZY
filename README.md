
# MAIA
MAIA est un logiciel de reconaissance de signalisation routiere. Développé en Java et exécutable via Maven.

## Fonctionnalités:                                                                                                                                                                                                                                                            
L’application permet de lancer une reconnaissance de panneaux routiers :

- sur une image enregistrée sur votre ordinateur ;
- sur une vidéo enregistrée sur votre ordinateur ;
- en temps réel via votre webcam.


## Modes de détection


L’application propose trois modes de détection :

- un réseau de neurones entraîné en Python puis exporté au format ONNX pour être utilisé en Java ;
- un réseau de neurones entièrement conçu par nos développeurs avec DL4J (DeepLearning4J) ;
- une application externe en Python utilisant YOLOv8. Cette dernière dispose de sa propre interface : la fenêtre Java se fermera donc automatiquement lors de son lancement.


## Pour Commencer  


Au démarrage, une image cartoon des développeurs s’affiche.  
Pour accéder à l’application, cliquez sur le bouton **« Entrer dans l’application »** caché sur l’ordinateur sur le bureau.

Vous devrez ensuite :

1. sélectionner le modèle de détection souhaité ;
2. choisir le mode d’acquisition ;
3. lancer la reconnaissance.
### Prérequis:                                                                                                                                                                                                                                                                     
- Java 1.8                                                                                                                                                                                                                                                                     
- Maven 4.0                                                                                                                                                                                                                                                                     
- OpenCV 2.4.13

Pour vérifier, faites ces commandes dans votre invite de commande :  
- java -version  
- mvn -version

### Installation:  
Clonez le projet et mettez vous dans le dossier 
```
git clone git@github.com:Ibtissam567/TWIZZY.git  
```
Ajoutez
Dans 
```
Java Build Path
``` 
Ajoutez le fichier à l'adresse 
*Adresse OpenCV* ```opencv\build\java\opencv-2413.jar```
Puis modifiez
```
Native library location
```
par

*Adresse OpenCV* ```opencv\build\java\x64```

### Démarrage  
  - Compiler le projet avec Maven : _mvn clean install_ 

  - Lancer le jeu : _mvn exec:java_ 

## Structure du projet :

```bash
├──.settings
├──backlog
├──bin
├── src
│   ├── main
│   │   ├── java
│   │   │   ├── com
│   │   │   │   ├── logiciel
│   │   │   │   │   ├── Main.java
│   │   ├── resources
│   └── test
│   │   ├── java
│   │   │   ├── com
│   │   │   │   ├── logiciel
│   │
├── target 
├── .classpath
├── .project
├── pom.xml
├── .gitignore
├── local.properties
└── README.md
```
