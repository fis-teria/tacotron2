import pyopenjtalk
text = "システィナ礼拝堂は、１４７３年に、バティカン宮殿内に建立された、壮大な礼拝堂です、"
phones = pyopenjtalk.g2p(text, kana=False)
print(phones)
