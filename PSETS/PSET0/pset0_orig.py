# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 0: Python Coding Questions - Fall 2023
Due September 13, 2023 at 11:59pm
"""

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset0.py" exactly
# 3. No outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
# Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


###################################
# Question 3.1: Pascal's Triangle #
###################################

# Instructions: Write an iterative algorithm to generate the 0th to nth rows of Pascal's triangle where indexing starts at 0. 
# E.g. the triangle corresponding to n=5 will have 6 rows in total, one for the 0th row, the 1st row, etc. up until the 
# 5th row. Complete the function below and test your function against the provided test cases to make sure that it is working.
# Do not use any external libraries or online solutions, please use only base python functions and data types.
# More info about Pascal's Triangle can be found here: https://en.wikipedia.org/wiki/Pascal%27s_triangle

# Complete the function below and test your function against the provided test cases.
# Do not use any external libraries, please use only base python functions and objects.

# Input: An integer n>=0 denoting the output triangle's depth, indexing starts at 0
# Output: A list of lists containing the first n rows of Pascal’s Triangle

# Example 1: If n = 0 then the return [1]
# Example 2: If n = 3 then the return [[1],[1,1],[1,2,1],[1,3,3,1]]


def pascals_triangle(n:int)->list:
    """Returns as a list of lists containing the 0 to n rows of Pascal's Triangle"""
    #### YOUR CODE HERE ####
    pass


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    assert pascals_triangle(0) == [[1]] # Remember, you must return a list of lists!
    assert pascals_triangle(3) == [[1],[1,1],[1,2,1],[1,3,3,1]]
    assert pascals_triangle(10) == [[1],
                                     [1, 1],
                                     [1, 2, 1],
                                     [1, 3, 3, 1],
                                     [1, 4, 6, 4, 1],
                                     [1, 5, 10, 10, 5, 1],
                                     [1, 6, 15, 20, 15, 6, 1],
                                     [1, 7, 21, 35, 35, 21, 7, 1],
                                     [1, 8, 28, 56, 70, 56, 28, 8, 1],
                                     [1, 9, 36, 84, 126, 126, 84, 36, 9, 1],
                                     [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]]
    print("All sample test cases for pascals_triangle passed!")

# Helper function DO NOT EDIT
def print_pascals_triangle(triangle_rows):
    """This function prints out a Pascal Triangle list of lists to the console in a readable format"""
    largest_length=len(str(triangle_rows[-1]))
    for e in range(0,len(triangle_rows)):
        current_row_length=len(str(triangle_rows[e]))
        print("["+str(e)+"]",end="")
        for i in range(0,(largest_length-current_row_length)//2):
            print(" ",end="")
        print(triangle_rows[e])

# Print Pascal's Triangle in a visually appealing format
if __name__ == "__main__":
    print_pascals_triangle(pascals_triangle(3))
    print_pascals_triangle(pascals_triangle(5))
    print_pascals_triangle(pascals_triangle(10))



##################################
# Question 3.2: Flatten Function #
##################################

# Instructions: Write a recursive algorithm that outputs a list of integers which are extracted from a nested set 
# of iterable lists. Your function should flatten a nested structure of integers contained in various lists into a 
# simple list of integers. For example if the input is [1,2,[3,4],[[5],6,7]] then return [1,2,3,4,5,6,7]. Please be 
# sure to use recursion for this exercise. 

# Complete the function below and test your function against the provided test cases.
# Do not use any external libraries, please use only base python functions and data types.

# Input: A nested structure built as a list of lists and/or integers
# Output: A list of all integers contained in the nested iterable list structure

# Example 1: If input_list = [1,2,[3,4],[[5],6,7]] then the return [1,2,3,4,5,6,7]
# Example 2: If input_list = [[1,2],[[3]],5,6] then return [1,2,3,4,5,6,7]

def flatten(input_list:list)->list:
    """Recursive function that extracts all leaf node integers from a nested structure of lists as 1 output list"""
    #### YOUR CODE HERE ####
    pass


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    assert flatten([1,2,[3,4],[[5],6,7]]) == [1, 2, 3, 4, 5, 6, 7]
    assert flatten([[1,2],[[3]],5,6]) == [1, 2, 3, 5, 6]
    assert flatten([2,4,6]) == [2, 4, 6]
    assert flatten([]) == []
    assert flatten([[[[[[[[4]]]]]]]]) == [4]
    print("All sample test cases for flatten passed!")





###############################
# Question 4.1: Caesar Cipher #
###############################

# Instructions: Complete the class declaration in below to create a custom, user-defined data structure that can save 
# a cipherkey, an alphabet, and also perform encryption and decryption of a message using a Caesar Cipher. A Caesar Cipher 
# is a simple encryption method that shifts all letters by a fixed offset specified by the cipherkey's location in the 
# alphabet. E.g. if our alphabet is the standard lower-case English letters abc...xyz and our message is "x" and our 
# cipherkey is "b" then the resulting ciphertext would be "y" since the cipherkey "b" corresponds to the index 1 letter 
# of the alphabet so we offset the plaintext message "x" by +1 letter to get "y". If our cipherkey was "d", we would 
# offset by 3 since "d" is at index 3 in the alphabet (indexing from 0) which would result in our message being encrypted 
# to "a". We wrap around to the start of the alphabet if our offset exceeds the last letter. Likewise, for decryption, 
# instead of offsetting forward, we offset backwards to decrypt a ciphertext into plaintext. See the diagram in the pdf
# instructions guide for a visual. Note, an alphabet is any list of string characters provided to the class, it need not
# be restricted to the default abc...xyz, your code should be robust to operate on any arbitrary alphabet provided.

# Do not change the names of the methods or class. Do not use external libraries of data structures. 

# Your code should include the following methods:

# __init__: Write a constructor method that takes 2 input arguments specifying the alphabet to use and a cipherkey.
# Both should be stored as attributes. The default alphabet if none is provided should be the 26 lower-case letters
# of the English alphabet i.e. abc...xyz and None as a default for the cipherkey.

# encrypt: Write a method that takes in a plaintext string to be encrypted and a string that designates the key
# from the alphabet to use as the cipherkey. If the cipherkey is not an element of the alphabet, raise an AttributeError.
# If no key provided, default to the key stored from the initialization. If an input character of the plaintext is not 
# in the alphabet, do not alter it, return it as is. 
# Input: plaintext string and cipherkey string. Output: ciphertext string

# decrypt: Write a method that takes in a ciphertext string to be decrypted and a string that designates the key
# from the alphabet to use as the cipherkey. If the cipherkey is not an element of the alphabet, raise an AttributeError. 
# If no key provided, default to the key stored from the initialization. If an input character of the ciphertext is not 
# in the alphabet, do not alter it, return it as is. 
# Input: ciphertext string. Output: plaintext string


###############################################
# Question 4.2: Caesar Cipher Auto-Decryption #
###############################################

# Instructions: Add to the CaesarCipher class a new method called auto_decrypt() that will automatically try to guess the 
# correct cipherkey of the message by trying each character in the saved alphabet and computing the % of words recognized in 
# the English language contained in the resulting plaintext. Return a tuple containing the decrypted plaintext and  
# auto-detected cipherkey. You may add early stopping and stop if you find a cipherkey that produces a plaintext output with  
# at least 85% words recognized in the English word dictionary. If no key produces at least 85% recognizable words, then return 
# the key with the highest % of recognized words. The english_word_list has been read in for you, pass in this object by 
# reference into your method call. Add an optional argument called verbose with a default of False that will print out the 
# % of words recognized as English words for each character tried as cipherkey to track the progress of auto-detection.

# Note, all words in the  english_word_list are in lower case, you should make comparisons in lower case for each word. 
# Input: ciphertext string. Output: tuple containing the plaintext string and the detected cipherkey

### Bonus Problem: Can you think of a faster way to auto-decrypt a secret message (in English) that was encrypted using a Caesar 
# cipher? Briefly describe your approach and add an additional method to CaesarCipher below called auto_decrypt_bonus() if you're 
# up for the challenge. The top 5 fastest submissions by average runtime on a set of hidden test cases that return the correct 
# output for all test cases will earn extra credit on this assignment. The class object is instantiated once at the start and all 
# test cases are run on it. Assume the alphabet used for this question is the list of standard English lower case letters i.e. abc...xyz. 
# This implies also that all input letters of the ciphertext will be lower case English letters. Do not use any external libraries, 
# please use only base python functions and objects.

# Hint: A table containing the average frequency of use for each letter in the alphabet in written English text has also been 
# provided that you can utilize in your approach. 

# Input: ciphertext string. Output: tuple containing the plaintext string and the detected cipherkey (same as Part 2a)

### Part 1 and 2: Caesar Cipher ###

class CaesarCipher:
    def __init__(self, alphabet=list('abcdefghijklmnopqrstuvwxyz'), cipherkey=None):
        #### YOUR CODE HERE ####
        pass

    def encrypt(self, plaintext:str, cipherkey:str = None)->str:
        """Convert plaintext to ciphertext using a cipherkey"""
        #### YOUR CODE HERE ####
        pass
    
    def decrypt(self, ciphertext:str, cipherkey:str = None)->str:
        """Convert ciphertext to plaintext using a cipherkey"""
        #### YOUR CODE HERE ####
        pass
    
    def auto_decrypt(self, ciphertext:str, english_word_list:list, verbose=False)->tuple:
        """Decrypt a ciphertext by iteratively searching for the most likely cipherkey"""
        #### YOUR CODE HERE ####
        pass
    
    def auto_decrypt_bonus(self, ciphertext:str, english_letter_freq_table:dict)->tuple:
        """Bonus Problem: A faster, more efficient method of auto-decrypting a Caesar Cipher"""
        #### YOUR CODE HERE ####
        pass
    

# Part 2: English dictionary word list
english_word_list = open("english_words_list.txt",'r').readlines()
english_word_list = [word.rstrip("\n") for word in english_word_list]

# Bonus Question: English language letter frequency table
english_letter_freq_table = open("letter_freq_table.txt",'r').readlines()
english_letter_freq_table = [tuple(row.rstrip("\n").split("\t")) for row in english_letter_freq_table]
english_letter_freq_table = {letter:float(freq) for letter,freq in english_letter_freq_table}


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    test_cipher = CaesarCipher()
    ciphertext_test = 'nkrru gtj ckriusk zu iusvazkx yioktik utk komnze zcu! Il eua gxk ghrk zu xkgj znoy ykixkz skyygmk, zngz skgty euax vxumxgs oy cuxqotm.'
    print(test_cipher.decrypt(ciphertext_test,'g')) # Should print out a readable message
    plaintext_test = 'This is a plaintext message to be encrypted using your algorithm!'
    assert test_cipher.encrypt(plaintext_test, 'a') == plaintext_test # 'a' corresponds to no shifting of letters
    assert test_cipher.decrypt(test_cipher.encrypt(plaintext_test, 'q'),'q') == plaintext_test # Decrypt should reverse encrypt
    
    test_plain_txt = "sometimes text can have misspelings in it or names like alice, bradley, cassandra, or diego"
    test_cipher_txt = 'pljbqfjbp qbuq zxk exsb jfppmbifkdp fk fq lo kxjbp ifhb xifzb, yoxaibv, zxppxkaox, lo afbdl'
    assert test_cipher.encrypt(test_plain_txt,"x") == test_cipher_txt
    assert test_cipher.auto_decrypt(test_cipher_txt, english_word_list, verbose=True) == (test_plain_txt, 'x')
    print("All sample test cases for CaesarCipher passed!")

# Bonus question sample-test case  - uncomment to run
#assert test_cipher.auto_decrypt_bonus(test_cipher_txt, english_letter_freq_table) == (test_plain_txt, 'x')


#################################
# Question 4.3: Vigenère Cipher #
#################################

# Instructions: Implement a class-based Vigenère Cipher by completing the starter code for VigenereCipher below. Your data 
# structure should have a method for encryption and decryption. A Vigenère cipher can be thought of as a generalization of 
# a Caesar cipher. Vigenère ciphers convert plaintext to ciphertext using a cipherkey that may be of length 1 or greater 
# where each subsequent character in the cipherkey specifies the alphabetical offset for each subsequent plaintext letter 
# being encrypted. The cipherkey is resued from start to end as many times as needed to encrypt or decrypt the entire message. 
# A Caesar cipher is a special case of a Vigenère cipher where the cipherkey is of length 1. 

# Note, an alphabet is any list of string characters provided to the class, it need not be restricted to the default abc...xyz,
#  your code should be robust to operate on any arbitrary alphabet provided. Do not use any external libraries. 

# Your code should include the following methods:

# __init__: Write a constructor method that takes 2 input arguments specifying the alphabet to use and a cipherkey.
# Both should be stored as attributes. The default alphabet if none is provided should be the 26 lower-case letters
# of the English alphabet i.e. abc...xyz and None as a default for the cipherkey.

# encrypt: Write a method that takes in a plaintext string to be encrypted and a string that designates the cipherkey
# containing characters from the saved alphabet. If a character in the cipherkey is not an element of the alphabet, raise 
# an AttributeError. If no key provided, default to the key stored from the initialization. If an input character of the 
# plaintext is not in the saved alphabet, do not alter it, return it as is. 
# Input: plaintext string and cipherkey string. Output: ciphertext string

# decrypt: Write a method that takes in a ciphertext string to be decrypted and a string that designates the cipherkey
# containing characters from the saved alphabet. If a character in the cipherkey is not an element of the alphabet, raise 
# an AttributeError. If no key provided, default to the key stored from the initialization. If an input character of the 
# ciphertext is not in the saved alphabet, do not alter it, return it as is. 
# Input: ciphertext string and cipherkey string. Output: plaintext string

# Hint: Consider using a Python generator to iteratively cycle through the letters of the cipherkey on loop.


class VigenereCipher:
    def __init__(self, alphabet=list('abcdefghijklmnopqrstuvwxyz'), cipherkey=None):
        #### YOUR CODE HERE ####
        pass

    def encrypt(self, plaintext:str, cipherkey:str = None)->str:
        """Convert plaintext to ciphertext using a cipherkey"""
        #### YOUR CODE HERE ####
        pass

    def decrypt(self, ciphertext:str, cipherkey:str = None)->str:
        """Convert ciphertext to plaintext using a cipherkey"""
        #### YOUR CODE HERE ####
        pass


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    alphabet_str = "abcdefghijklmnopqrstuvwxyz";cipherkey = "CatInTheHat"
    test_cipher = VigenereCipher(alphabet=list(alphabet_str+alphabet_str.upper()))
    test_plain_txt = "Encryption and decryption are interesting topics! Cryptography is an exciting field of study."
    test_cipher_txt = 'gnvZLiAmVn tPd wMpkFtaiHP aKM vgAiYeLViGO GhwmJs! VTyIbBZyeWhR Ks tV rqjmaiGI fBMyW vj ZtNFy.'
    assert test_cipher.encrypt(test_plain_txt,cipherkey) == test_cipher_txt
    assert test_cipher.decrypt(test_cipher_txt,cipherkey) == test_plain_txt
    print("All sample test cases for VigenereCipher passed!")
