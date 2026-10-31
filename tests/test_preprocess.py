from types import SimpleNamespace

from valeval.preprocess import SEP, UNK, Vocabulaire, tokeniser, tokens_argument


def test_negation_coupee_comme_glove():
    assert tokeniser("I don't, can't") == ["i", "do", "n't", ",", "ca", "n't"]


def test_possessif_garde_son_apostrophe():
    assert tokeniser("it's the child's") == ["it", "'s", "the", "child", "'s"]


def test_guillemet_ouvrant_retire():
    assert tokeniser("a 'good' idea") == ["a", "good", "idea"]


def test_ordre_conclusion_position_premisse():
    arg = SimpleNamespace(conclusion="Ban it", stance="against", premise="No.")
    assert tokens_argument(arg) == ["ban", "it", SEP, "against", SEP, "no", "."]


def test_vocabulaire_filtre_les_mots_rares():
    v = Vocabulaire([["a", "a", "b"]], min_freq=2)
    assert "a" in v.indice and "b" not in v.indice
    assert v.encoder(["a", "b"], 10) == [v.indice["a"], v.indice[UNK]]


def test_encoder_coupe_a_max_len():
    v = Vocabulaire([["a", "a"]], min_freq=1)
    assert len(v.encoder(["a"] * 5, 3)) == 3
