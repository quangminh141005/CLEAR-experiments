from importlib import resources

import clear


def test_package_import_exposes_supported_entry_points() -> None:
    assert callable(clear.load_pretrained)
    assert callable(clear.build_cxr_preprocess)
    assert clear.ConceptBank.__module__ == "clear.concepts"


def test_tokenizer_vocabulary_is_packaged() -> None:
    vocabulary = resources.files("clear").joinpath("bpe_simple_vocab_16e6.txt.gz")
    assert vocabulary.is_file()
