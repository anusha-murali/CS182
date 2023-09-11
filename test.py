def encrypt(plaintext:str, cipherkey:str = None)->str:
        """Convert plaintext to ciphertext using a cipherkey"""
        plaintext = ""
        offset = alphabet.index(cipherkey)
        for i in range(len(plaintext)):
            plainChar = plaintext[i]
            
            cipherChar = alphabet[(alphabet.index(plainChar) + offset) \
                                  % len(alphabet)]
            ciphertext = plaintext + cipherChar
        return ciphertext
                             
def decrypt(ciphertext:str, cipherkey:str = None)->str:
    
     """Convert ciphertext to plaintext using a cipherkey"""
     plaintext = ""
     offset = alphabet.index(cipherkey)


     for i in range(len(ciphertext)):
            cipherChar = ciphertext[i]
            plainChar = alphabet[(alphabet.index(cipherChar) - offset) \
                                  % len(alphabet)]
            plaintext = plaintext + plainChar
     return plaintext
    
    
