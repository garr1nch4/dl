import string
from random import randint
import csv


class CaesarCipher:

    def __init__(self, alphabet: str, shift: int):
        if shift >= len(alphabet) or shift <= 0:
            raise Exception('Invalid shift size')
        encode_alphabet = alphabet[shift:] + alphabet[:shift]
        self.__encode_dict = dict(zip(alphabet, encode_alphabet))
        self.__decode_dict = dict(zip(encode_alphabet, alphabet))

    def encode(self, phrase: str):
        return ''.join((self.__encode_dict[key] for key in phrase if key in self.__encode_dict))

    def decode(self, phrase: str):
        return ''.join((self.__decode_dict[key] for key in phrase if key in self.__decode_dict))


class RandomLetterPhraseGenerator:

    def __init__(self, alphabet: str, min_length: int = 10, max_length: int = 20):
        self.alphabet = alphabet
        self.min_length = min_length
        self.max_length = max_length

    def make(self):
        phrase_len = randint(self.min_length, self.max_length) + 1
        alphabet_len = len(self.alphabet) - 1
        return ''.join((self.alphabet[randint(0, alphabet_len)] for _ in range(phrase_len)))


def main():
    alphabet = string.ascii_lowercase + ' '
    cipher = CaesarCipher(alphabet, shift=3)
    generator = RandomLetterPhraseGenerator(alphabet)
    with open('dataset.csv', 'w') as file:
        writer = csv.writer(file)

        for i in range(20000):
            e = cipher.encode(generator.make())
            writer.writerow([e, cipher.decode(e)])


if __name__ == '__main__':
    main()