**Description du projet**

Ce projet a pour but de simuler les outils utilisés par une équipe de recherche quantitative. Nous avons développé une application web en Python avec la librairie Streamlit, hébergée en continu sur une machine virtuelle Linux. L'objectif était de combiner l'analyse technique d'actifs individuels et la gestion de portefeuille dans une interface unique, capable de fonctionner en temps réel.
L'application est divisée en deux modules principaux correspondant à la répartition des tâches au sein du binôme.

**Module Quant A (Tomas)**

Ce module se concentre sur l'analyse d'un seul actif à la fois, qu'il s'agisse d'une action ou d'une crypto-monnaie. La particularité technique de cette partie réside dans la gestion intelligente des données : le code adapte automatiquement l'intervalle de téléchargement selon la période choisie. Si l'utilisateur observe une période courte comme 1 ou 5 jours, l'application télécharge des données toutes les 5 minutes. Pour des périodes plus longues, elle bascule sur des données journalières.
Au niveau des stratégies, nous avons implémenté un croisement de moyennes mobiles (SMA) avec une option d'optimisation automatique qui teste différentes combinaisons pour trouver la plus performante sur le passé. Une stratégie basée sur le RSI est également disponible pour détecter les zones de surachat et de survente. Enfin, nous avons ajouté une fonctionnalité de prédiction utilisant une régression linéaire (Machine Learning) pour estimer la tendance du prix sur les prochains jours avec un intervalle de confiance.

**Module Quant B (Mehdi)**

Ce module permet de construire et de suivre un portefeuille composé de plusieurs actifs. Nous avons voulu rendre ce simulateur utilisable à la fois pour du backtesting long terme et du suivi en direct. L'utilisateur peut choisir entre un mode Historique sur 2 ans et un mode Live sur les 5 derniers jours. Dans ce mode Live, le graphique se rafraîchit automatiquement toutes les 5 minutes pour suivre l'évolution du marché.
L'utilisateur peut définir la pondération de chaque actif soit de manière équipondérée, soit manuellement. Le simulateur compare ensuite deux approches de gestion : une stratégie passive (Buy and Hold) et une stratégie active avec rééquilibrage quotidien (Constant Mix). Les indicateurs de performance comme le ratio de Sharpe, la volatilité et la matrice de corrélation sont calculés instantanément.

**Infrastructure et Déploiement Continu (AWS)**

Pour que l'application soit accessible 24h/24 sans dépendre de nos ordinateurs personnels, nous l'avons déployée sur une instance AWS tournant sous Ubuntu.
Le défi principal était de maintenir l'application active même après la fermeture de la connexion SSH. Pour résoudre cela, nous avons utilisé Tmux. Tmux permet de créer une session virtuelle persistante sur le serveur. Concrètement, nous lançons l'application Streamlit à l'intérieur de cette session, puis nous partons de la session. Le processus continue ainsi de tourner en arrière-plan indéfiniment, garantissant la disponibilité du site.

**Automatisation et Rapports**

Pour répondre aux exigences de reporting, nous avons mis en place un script Python autonome nommé report.py. Ce script est exécuté automatiquement chaque jour à 20h00 par le gestionnaire de tâches cron du serveur.
Concrètement, ce script télécharge les données de l'action de référence (Engie), calcule la volatilité journalière ainsi que le Max Drawdown de la journée en cours, et sauvegarde ces informations dans un fichier CSV sur le serveur.

**Lancer le projet**
Pour le projet, il faut récupérer sur notre dossier Github le fichier portfolio.py(pour tout le site), le fichier report.py, avoir une clé AWS et un dossier où mettre les rapports journaliers. Pas besoin de récupérer le fichier app.py, il servait pour la partie Quant A mais a été complètement intégré à portfolio.py nous avons laissé app.py dans le main juste pour montrer le merge de la partie Quant A.
