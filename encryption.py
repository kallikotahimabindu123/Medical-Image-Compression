from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import pickle

key = get_random_bytes(32)

def encrypt(data):

    cipher=AES.new(key,AES.MODE_EAX)

    ciphertext,tag=cipher.encrypt_and_digest(pickle.dumps(data))

    return cipher.nonce,ciphertext,tag


def decrypt(nonce,ciphertext,tag):

    cipher=AES.new(key,AES.MODE_EAX,nonce=nonce)

    data=pickle.loads(cipher.decrypt(ciphertext))

    return data