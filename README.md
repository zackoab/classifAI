## Système de Classification Avancé assisté par IA

### Contexte et Objectif

Ce test technique vise à évaluer votre capacité à concevoir et implémenter un système de classification innovant, exploitant les technologies d'IA modernes et particulièrement les LLM. Le défi consiste à créer un système capable de différencier deux catégories très subtiles, tout en étant facile à utiliser et transparent dans son fonctionnement. L'accent est mis sur la créativité et l'exploration de méthodes d'utilisation des LLM pour obtenir le meilleur système possible.

### Contraintes et Exigences

### 1. Facilité de configuration et d'utilisation

- Le système doit minimiser l'effort de prompting pour l'utilisateur final
- **L'utilisateur final ne doit pas avoir besoin d'être initié au prompt engineering ou de posséder des connaissances techniques spécifiques aux LLM**
- Le système doit rendre la classification accessible aux non-experts, sans nécessiter d'apprentissage spécifique
- Les fichiers de configuration doivent être simples et bien conçus (une interface est un plus apprécié mais non obligatoire)
- La configuration initiale doit être simple et rapide à mettre en place
- Toute idée créative facilitant davantage l'utilisation est encouragée

### 2. Transparence du processus de classification

- Le système doit expliquer clairement les raisons de ses décisions de classification
- Les facteurs déterminants dans la classification doivent être mis en évidence
- D'autres approches innovantes pour rendre le système transparent sont bienvenues

### 3. Système d'évaluation assisté par IA

- Développer des mécanismes automatisés pour valider les résultats
- Concevoir un système capable de générer et tester différents cas de figure sans dataset préexistant
- Implémenter des méthodes d'auto-évaluation et d'amélioration continue
- Toute approche créative permettant une meilleure évaluation sera valorisée

### Critères d'Évaluation

Votre solution sera évaluée selon les critères suivants:

1. **Qualité en ML du système**
    - Efficacité du système de classification pour des catégories subtiles
    - Robustesse et fiabilité des prédictions
    - Pertinence des méthodes d'évaluation implémentées
2. **Créativité et innovation**
    - Originalité dans l'utilisation des LLM
    - Approches innovantes pour résoudre les défis de classification subtile
    - Solutions créatives pour l'évaluation sans dataset
3. **Propreté et qualité du code**
    - Structure et organisation du code
    - Documentation et lisibilité
    - Bonnes pratiques de développement

### Conseils pour Exceller

- Concentrez-vous sur l'exploration créative des capacités des LLM
- Pensez à des approches non conventionnelles pour la classification de catégories subtiles
- Développez des méthodes originales pour l'auto-évaluation du système
- Privilégiez la simplicité d'utilisation sans sacrifier les performances
- Exploitez les capacités génératives des modèles d'IA pour compenser l'absence de dataset


---------------------------------------

### Utilisation

Pour utiliser ce projet, il suffit de lancer le fichier notebook.ipynb et de suivre les informations demandées.
Les modèles de LLM intégrés sont gpt-4o et gpt-4o-mini.<br>
Les packages à installer sont indiqués dans le fichier requirements.txt.

Pour évaluer le modèle à partir d'un fichier .txt, le format doit être le suivant :<br>
<br>
[verbatim 1] // [label 1]<br>
[verbatim 2] // [label 2]<br>
...<br>
Un exemple est donné dans le fichier validation.txt.<br>

Pour tester le modèle à partir d'un fichier .txt, le format doit être le suivant :

[verbatim 1]<br>
[verbatim 2]<br>
...<br>
Un exemple est donné dans le fichier test.txt.

Si le fichier d'évaluation/de test n'est pas spécifié, le modèle propose de générer un dataset synthétique.<br>
La partie "évaluation" du notebook affiche les metriques de performance de la classification (accuracy, precision, recall, f1-score, confusion matrix).<br>
La partie "utilisation" du notebook affiche les prédictions du modèle pour chaque verbatim du fichier de test, ainsi que les raisons du choix du modèle.<br>

Cette version du projet est assez basique. Propositions d'amélioration :
- intégrer d'autres modèles de LLM
- ajouter des fonctions de validation des datasets d'évaluation/test générés