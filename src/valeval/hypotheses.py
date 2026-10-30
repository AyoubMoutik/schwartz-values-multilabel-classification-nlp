"""Partie B1 : descriptions des 20 valeurs, en 3 formulations (d'après value-categories.json)."""

from valeval.data import VALUES

# Formulation 1 : anglais, description de la valeur
EN = {
    "Self-direction: thought": "This argument is about thinking for oneself and being free to form one's own opinions.",
    "Self-direction: action": "This argument is about being free to choose one's own actions and goals.",
    "Stimulation": "This argument is about excitement, novelty and a varied life.",
    "Hedonism": "This argument is about pleasure and enjoying life.",
    "Achievement": "This argument is about success, ambition and being competent.",
    "Power: dominance": "This argument is about having power and control over people.",
    "Power: resources": "This argument is about wealth and control over material resources.",
    "Face": "This argument is about protecting one's public image and avoiding humiliation.",
    "Security: personal": "This argument is about personal health, safety and well-being.",
    "Security: societal": "This argument is about order, stability and safety in society.",
    "Tradition": "This argument is about respecting traditions, customs and religious faith.",
    "Conformity: rules": "This argument is about obeying laws, rules and obligations.",
    "Conformity: interpersonal": "This argument is about being polite and avoiding upsetting other people.",
    "Humility": "This argument is about being humble and accepting one's modest place.",
    "Benevolence: caring": "This argument is about caring for the well-being of family and friends.",
    "Benevolence: dependability": "This argument is about being a loyal and reliable member of one's group.",
    "Universalism: concern": "This argument is about equality, justice and protecting all people.",
    "Universalism: nature": "This argument is about protecting nature and the environment.",
    "Universalism: tolerance": "This argument is about accepting and understanding people who are different.",
    "Universalism: objectivity": "This argument is about seeking the truth and thinking rationally without bias.",
}

# Formulation 2 : français, traduction de la formulation 1
FR = {
    "Self-direction: thought": "Cet argument porte sur le fait de penser par soi-même et d'être libre de se forger ses propres opinions.",
    "Self-direction: action": "Cet argument porte sur la liberté de choisir ses propres actions et ses propres objectifs.",
    "Stimulation": "Cet argument porte sur l'excitation, la nouveauté et une vie variée.",
    "Hedonism": "Cet argument porte sur le plaisir et le fait de profiter de la vie.",
    "Achievement": "Cet argument porte sur la réussite, l'ambition et la compétence.",
    "Power: dominance": "Cet argument porte sur le pouvoir et le contrôle exercés sur les personnes.",
    "Power: resources": "Cet argument porte sur la richesse et le contrôle des ressources matérielles.",
    "Face": "Cet argument porte sur la protection de son image publique et le fait d'éviter l'humiliation.",
    "Security: personal": "Cet argument porte sur la santé, la sécurité et le bien-être personnels.",
    "Security: societal": "Cet argument porte sur l'ordre, la stabilité et la sécurité dans la société.",
    "Tradition": "Cet argument porte sur le respect des traditions, des coutumes et de la foi religieuse.",
    "Conformity: rules": "Cet argument porte sur l'obéissance aux lois, aux règles et aux obligations.",
    "Conformity: interpersonal": "Cet argument porte sur la politesse et le fait d'éviter de contrarier les autres.",
    "Humility": "Cet argument porte sur l'humilité et le fait d'accepter sa modeste place.",
    "Benevolence: caring": "Cet argument porte sur le souci du bien-être de sa famille et de ses amis.",
    "Benevolence: dependability": "Cet argument porte sur le fait d'être un membre loyal et fiable de son groupe.",
    "Universalism: concern": "Cet argument porte sur l'égalité, la justice et la protection de tous.",
    "Universalism: nature": "Cet argument porte sur la protection de la nature et de l'environnement.",
    "Universalism: tolerance": "Cet argument porte sur l'acceptation et la compréhension des personnes différentes.",
    "Universalism: objectivity": "Cet argument porte sur la recherche de la vérité et la pensée rationnelle, sans parti pris.",
}

# Formulation 3 : reformulation centrée sur l'auteur et sur ce qu'il défend
REF = {
    "Self-direction: thought": "The author believes people should be able to make up their own minds.",
    "Self-direction: action": "The author believes people should decide for themselves how to live.",
    "Stimulation": "The author wants life to be exciting and full of new experiences.",
    "Hedonism": "The author wants people to have fun and enjoy themselves.",
    "Achievement": "The author wants people to succeed and be recognized for their abilities.",
    "Power: dominance": "The author thinks it matters who has authority over others.",
    "Power: resources": "The author thinks money and property matter.",
    "Face": "The author cares about reputation and not losing face.",
    "Security: personal": "The author wants individuals to be healthy and safe from harm.",
    "Security: societal": "The author wants the nation and society to remain safe and stable.",
    "Tradition": "The author wants to preserve customs, heritage and religion.",
    "Conformity: rules": "The author thinks everyone should follow the rules.",
    "Conformity: interpersonal": "The author thinks we should not offend or bother others.",
    "Humility": "The author thinks people should not put themselves first.",
    "Benevolence: caring": "The author wants to help the people close to us.",
    "Benevolence: dependability": "The author thinks people should be trustworthy and keep their commitments.",
    "Universalism: concern": "The author wants everyone to be treated fairly and to have equal rights.",
    "Universalism: nature": "The author wants to preserve the planet and its wildlife.",
    "Universalism: tolerance": "The author wants different people and cultures to be accepted.",
    "Universalism: objectivity": "The author wants decisions to be based on facts and evidence.",
}

FORMULATIONS = {"en": EN, "fr": FR, "reformulation": REF}


def liste(formulation: str) -> list:
    """Les 20 hypothèses, dans l'ordre de VALUES."""
    return [FORMULATIONS[formulation][v] for v in VALUES]
