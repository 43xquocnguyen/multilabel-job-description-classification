from setup import *
from unidecode import unidecode

# Check if element is word and '.' or not
def is_word(element):
    if element == '.':
        return True
    pattern = r'^\w+$'
    return re.match(pattern, element) is not None

# Punctuation & stopwords removal
def remove_characters(tokens):
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS and is_word(unidecode(token)):
                cleaned_tokens.append(token)
    return cleaned_tokens

# Preprocess function
def preprocess(txt):
    # Lower case
    txt = txt.lower()
    # Tokenization
    # txt = vncorenlp.tokenize(txt)
    # txt = list(chain.from_iterable(txt))
    txt = ViTokenizer.tokenize(txt).split()
    # Punctuation & stopwords removal
    txt = remove_characters(txt)
    txt = list(filter(lambda x: x != '', txt))
    # Join
    txt = ' '.join(txt)
    txt = txt.replace('_', ' ')
    txt = txt.replace(' . ', '. ')
    return txt

