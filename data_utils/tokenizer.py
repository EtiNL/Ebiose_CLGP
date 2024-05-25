from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing


# Function to train a Unigram tokenizer
def train_unigram_tokenizer(corpus, vocab_size=3000):
    # Initialize a Unigram model
    tokenizer = Tokenizer(models.Unigram())
    
    # Set the normalizer and pre-tokenizer
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    # Set up the trainer for Unigram
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    # Train the tokenizer
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    # Set post-processing template to handle special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    return tokenizer

def get_trainned_tokenizer(path):
    return Tokenizer.from_file(path)
    
if __name__=='__main__':
    # Sample corpus
    corpus = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "How about you?",
        "I'm good too, thanks for asking."
    ]
    # Train the tokenizer on the corpus
    tokenizer = train_unigram_tokenizer(corpus)

    # Save the tokenizer to disk
    tokenizer.save("unigram_tokenizer.json")

    # Load the tokenizer from disk
    loaded_tokenizer = Tokenizer.from_file("unigram_tokenizer.json")

    # Test the tokenizer
    test_sentence = "Hello, how are you?"
    encoded = loaded_tokenizer.encode(test_sentence)

    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)

    # Decode the token IDs back to a string
    decoded = loaded_tokenizer.decode(encoded.ids)
    print("Decoded:", decoded)
