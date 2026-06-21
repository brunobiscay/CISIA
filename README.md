


# NOTE BRUNO
1. On travaille sur le modele IA/machine learning
Une fois qqu'on aura le modele , le type de ML on passe à l'appli
- analyse RGPD et merise des données

CONTEXTE: je  souhaite développer un modèle IA pour aider dans à améliorer l'engorgement des services d'urgence (le développement de la télémédecine imposent des solutions de tri rapide et fiables). Ce projet consiste à concevoir un système d'intelligence artificielle capable de classer le degré d'urgence d'une situation à partir de données hybrides (numériques et textuelles).
Pour créer ce système d’IA, on a un jeu de données composé de données collectées au sein d’un établissement médical. Il contient 2000 échantillons pour lesquelles nous connaissons le niveau d’urgence (c’est la décision prise par le professionnel de santé à la lecture des informations).
Dans le détail, chaque échantillon est composé de :
● Données tabulaires : Âge, données administratives, constantes vitales (fréquence cardiaque, tension, température, saturation), antécédents médicaux (pathologie chronique) et durée des symptômes.
● Données textuelles : Description libre rédigée par le patient ou rapport d'appel du régulateur détaillant la plainte principale.
● Variable cible : Niveau d'urgence (0 : non urgent, 1 : urgence relative, 2 : urgence vitale).
L’enjeu pour vous de de réussir à exploiter ces données pour prédire si une situation rencontrée relève de l’urgence vitale, d’une urgence relative, ou au contraire ne contient aucun caractère urgent. Vous devrez pour ceci respecter les contraintes réglementaires et éthiques liées à l’usage de données personnelles sensibles. Vous devrez aussi adapter vos apprentissages afin de limiter le nombre de mauvaises classifications dangereuses d’un point de vue métier.


Description des champs de données

patient_id=Identifiant unique du patient.
sexe= Sexe du patient : F (Femme) ou H (Homme).
age=Âge du patient (en années).
zone_vie=Type de résidence : U (Urbain) ou R (Rural).
source=Origine de la donnée : appel (Transcription d'appel) ou chat (Interface texte).
freq_cardiaque=Fréquence cardiaque en battements par minute (bpm).
tension_sys=Tension artérielle systolique (mmHg).
temp=Température corporelle en degrés Celsius (°C).
sat_oxygene=Taux de saturation en oxygène (en %).
antecedents=Présence de pathologies chroniques : 1 (Oui), 0 (Non).
duree_symptomes=Temps écoulé depuis l'apparition des symptômes (en heures).
description_symptomes=Texte libre décrivant les douleurs, sensations ou l'état général.
niveau_urgence=Variable cible : 0 (Non urgent), 1 (Urgence relative), 2 (Urgence vitale).


Basé sur la methode MERISE peux tu concevoir les differentes tables de ma base de données et également me proposer de supprimer si besoin certains champs car non RGPD

2. Application dockerisée, CI/CD et monitorée



 Le modèle vise à aider des acteurs publics et privés à estimer la valeur de logements dans une logique d’aide à la décision.
Je dois concevoir une solution d’IA tout en prenant en compte un nouveau contexte d’usage, qui influencera l'identification des données pertinentes et la prise en compte les contraintes techniques, éthiques et réglementaires.
Ce geste s’inscrit dans une logique de démarche scientifique (formulation d’hypothèses, expérimentation, évaluation) tout en intégrant les enjeux métiers et sociétaux.