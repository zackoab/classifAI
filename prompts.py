def classification_prompt(concept: str, classe_1: str, classe_2: str) -> str:
    prompt = f"""
    - Contexte :
    Tu es expert en classification de verbatim, capable de différencier deux catégories très subtiles en te basant sur des éléments concrets et réfléchis.
    - Tâche :
    Classifier le verbatim donné en entrée en fonction de son : {concept}.
    Les deux classes possibles sont : "{classe_1}" et "{classe_2}". Tu dois uniquement renvoyer la classe correspondante.
    """
    return prompt


def evaluation_dataset_generation_prompt(
    concept: str, class_1: str, class_2: str, num_samples: int
) -> str:
    prompt = f"""
    - Contexte :
    Tu es un assistant spécialisé dans l'évaluation de modèle de classification de verbatim.
    - Tâche : 
    Tu dois générer un dataset pour évaluer un modèle de classification ayant pour critère de classification : {concept}.
    Les deux classes possibles pour les verbatim sont : "{class_1}" et "{class_2}".
    Génére un dataset de {num_samples} verbatim différents. Un verbatim est composé de 1 à 5 phrases.
    - Sortie :
    Chaque ligne en sortie est composée d'un verbatim suivi de " // " et de sa classe correspondante : "{class_1}" ou "{class_2}".
    Exemple :
    [verbatim 1] // [classe_1]
    [verbatim 2] // [classe_2]
    ...
    """
    return prompt


def test_dataset_generation_prompt(
    concept: str, class_1: str, class_2: str, num_samples: int
) -> str:
    prompt = f"""
    - Contexte :
    Tu es un assistant spécialisé dans l'évaluation de modèle de classification de verbatim.
    - Tâche : 
    Tu dois générer un dataset pour tester un modèle de classification ayant pour critère de classification : {concept}.
    Les deux classes possibles pour les verbatim sont : "{class_1}" et "{class_2}".
    Génére un dataset de {num_samples} verbatim différents. Un verbatim est composé de 1 à 5 phrases.
    - Sortie :
    Chaque ligne en sortie est composée d'un verbatim.
    Exemple :
    [verbatim 1]
    [verbatim 2]
    ...
    """
    return prompt
