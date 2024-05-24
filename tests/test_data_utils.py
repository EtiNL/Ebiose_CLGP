from tokenizers import Tokenizer

def test_tokenizer():
    # Load the tokenizer from disk
    loaded_tokenizer = Tokenizer.from_file("./data_utils/unigram_tokenizer.json")
    # Test the tokenizer
    test_sentence = "Hello, how are you?"
    encoded = loaded_tokenizer.encode(test_sentence)

    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)

    # Decode the token IDs back to a string
    decoded = loaded_tokenizer.decode(encoded.ids)
    assert encoded.tokens==decoded