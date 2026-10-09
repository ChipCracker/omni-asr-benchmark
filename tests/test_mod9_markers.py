from src.datasets.mod9_markers import clean_reference, is_blindtext


def test_marker_runs_and_always_markers_go():
    assert clean_reference("musik musik musik ja zu der ddrzeit war das") == "ja zu der ddrzeit war das"
    assert clean_reference("fremdsprache fremdsprache fremdsprache") == ""
    assert clean_reference("also leeresaudio leeres audio keinaudio kein audio fertig") == "also fertig"
    assert clean_reference("geräusch geräusch unverständlich und dann") == "und dann"


def test_spoken_words_survive_by_context():
    assert clean_reference("die musik war laut") == "die musik war laut"
    assert clean_reference("ich mache musik seit jahren") == "ich mache musik seit jahren"
    assert clean_reference("musik ist mein leben") == "musik ist mein leben"
    assert clean_reference("eine fremdsprache lernen") == "eine fremdsprache lernen"
    assert clean_reference("das war völlig unverständlich für mich") == "das war völlig unverständlich für mich"
    assert clean_reference("ein geräusch im keller") == "ein geräusch im keller"
    # Komposita sind keine Marker
    assert clean_reference("rockmusik und popmusik") == "rockmusik und popmusik"


def test_marker_inside_sentence_goes():
    assert clean_reference("wie ihr seht musik ist der platzbedarf kleiner") == "wie ihr seht musik ist der platzbedarf kleiner"
    assert clean_reference("anzupassen musik wie ihr seht") == "anzupassen wie ihr seht"
    assert clean_reference("im ergebnis dass heißt im unverständlich bereich") == "im ergebnis dass heißt im bereich"


def test_blindtext():
    assert is_blindtext("dies ist blindtext dies ist blindtext")
    assert not is_blindtext("blind ist der text")
