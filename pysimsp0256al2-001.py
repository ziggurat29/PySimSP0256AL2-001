#=========================================================================
#this script was created to test some SP-0256 related functionality.
#In particular:  
#Q:  can a reasonable approximation be made simply by appending
#pre-recorded phonemes?
#  A:  it is a little klunky, but it seems intelligible.
#Can my old text-to-speech code be ported and made to work?
#  A:  it seems so

#this was written assuming an Anaconda environment, but it should work fine
#in a conventional python environment provided you have the requisite packages.

import numpy as np
from scipy.io.wavfile import read, write
import pickle
#from IPython.display import Audio
#from numpy.fft import fft, ifft
#import matplotlib.pyplot as plt
#%matplotlib inline


#=========================================================================

#array of tuples of the SP-0256 phonemes; name, number, audio.
#note that we take a separate pass to load the audio from files named as
#per the phoneme from a well-known subdirectory.
#we subsequently build indices into this by name and by number.

'''
#note that the items marked YYY trigger an apparent bug in scipy/io/wavefile.py
#regarding 'incomplete chunk'.  I had to hack that file to work around the bug.
#because of this, I am going to pickle the collection, and provide that as an
#alternative method of loading the phoneme_list.
phoneme_list = [
[0x00,'PA1'],	#PAUSE		 1OMS
[0x01,'PA2'],	#PAUSE		 30MS
[0x02,'PA3'],	#PAUSE		 50MS
[0x03,'PA4'],	#PAUSE		1OOMS
[0x04,'PA5'],	#PAUSE		200MS
[0x05,'OY'],	#bOY		420MS
[0x06,'AY'],	#skY		260MS	#YYY bug in scipy/io/wavefile.py
[0x07,'EH'],	#End		 70MS
[0x08,'KK3'],	#Comb		120MS
[0x09,'PP'],	#Pow		21OMS
[0x0a,'JH'],	#doDGe		140MS
[0x0b,'NN1'],	#thiN		140MS
[0x0c,'IH'],	#sIt		 70MS
[0x0d,'TT2'],	#To			140MS	#YYY
[0x0e,'RR1'],	#Rural		170MS
[0x0f,'AX'],	#sUcceed	 70MS	#YYY
[0x10,'MM'],	#Milk		180MS
[0x11,'TT1'],	#parT		1OOMS
[0x12,'DH1'],	#THey		290MS	#YYY
[0x13,'IY'],	#sEE		250MS
[0x14,'EY'],	#bEIge		280MS
[0x15,'DD1'],	#coulD		 70MS
[0x16,'UW1'],	#tO			1OOMS
[0x17,'AO'],	#AUght		1OOMS
[0x18,'AA'],	#hOt		1OOMS
[0x19,'YY2'],	#Yes		180MS
[0x1a,'AE'],	#hAt		120MS
[0x1b,'HH1'],	#He			130MS
[0x1c,'BB1'],	#Business	 80MS
[0x1d,'TH'],	#THin		180MS
[0x1e,'UH'],	#bOOk		100MS
[0x1f,'UW2'],	#fOOd		260MS
[0x20,'AW'],	#OUt		370MS
[0x21,'DD2'],	#Do			160MS	#YYY
[0x22,'GG3'],	#wiG		140MS	#YYY
[0x23,'VV'],	#Vest		19OMS
[0x24,'GG1'],	#Got		 80MS
[0x25,'SH'],	#SHip		160MS
[0x26,'ZH'],	#aZure		190MS
[0x27,'RR2'],	#bRain		12OMS
[0x28,'FF'],	#Food		150MS	#YYY
[0x29,'KK2'],	#sKy		190MS	#YYY
[0x2a,'KK1'],	#Can't		160MS	#YYY
[0x2b,'ZZ'],	#Zoo		21OMS
[0x2c,'NG'],	#aNchor		220MS
[0x2d,'LL'],	#Lake		110MS
[0x2e,'WW'],	#Wool		180MS
[0x2f,'XR'],	#repAIR		360MS
[0x30,'WH'],	#WHig		200MS
[0x31,'YY1'],	#Yes		130MS	#YYY
[0x32,'CH'],	#CHurch		190MS
[0x33,'ER1'],	#fIR		160MS
[0x34,'ER2'],	#fIR		300MS	#YYY
[0x35,'OW'],	#bEAU		240MS	#YYY
[0x36,'DH2'],	#THey		240MS
[0x37,'SS'],	#veSt		 90MS
[0x38,'NN2'],	#No			190MS	#YYY
[0x39,'HH2'],	#Hoe		180MS	#YYY
[0x3a,'OR'],	#stORe		330MS
[0x3b,'AR'],	#alARm		290MS
[0x3c,'YR'],	#clEAR		350MS	#YYY
[0x3d,'GG2'],	#Guest		 40MS	#YYY
[0x3e,'EL'],	#saddLe		190MS
[0x3f,'BB2'],	#Business	 50MS	#YYY
]


print ( 'reading audio' )
#for each phoneme in 'phoneme_list', concoct a filename as
#  'allophones/' + name + '.wav'
#and load it.
for item in phoneme_list:
	filename = 'allophones/' + item[1].lower() + '.wav'
	#print(filename)
	fs,data = read(filename)	#fs is sampling rate, data is audio samples
	#the above will sometimes trigger an apparent bug in wavfile.py
	if len(data.shape) > 1:	#just keep the first channel if there are multiple
		data = data[:,0]
	item.append ( data )	#save der data at end


#pickling the collection to avoid the wavfile bug for others
with open ( "phonemes.bin", "wb" ) as fp:
	pickle.dump ( phoneme_list, fp )
'''

#unpickling the collection that was created as above
phoneme_list = []
with open ( "phonemes.bin", "rb" ) as fp:
	phoneme_list = pickle.load ( fp )


#now, make indices based on phoneme name and phoneme code.  indexing by code
#is useful when testing hand-crafted phoneme sequences.  indexing by symbolic
#name is useful when crafting text-to-speech rules.
phoneme_code = {}
phoneme_symb = {}
for item in phoneme_list:
	phoneme_code[item[0]] = item
	phoneme_symb[item[1]] = item


#print ( phoneme_code[0x30][1] )	#should be 'WH'
#print ( phoneme_symb['WH'][0] )	#should be 48


#=========================================================================


#whiz through a list of phoneme codes and dump the associated audio to make
#a sound file
def whizanddump ( filename, phonemeseq ):
	#for each phoneme in phonemeseq, lookup the audio and append it to the
	#output we are building
	output = np.empty((0), dtype=np.uint8)
	#print ( output )
	for val in phonemeseq:
		#print ( "for file: '" + filename + "', emitting: " + phoneme_code[val][1] + " (" + str(val) + ")" )
		output = np.append ( output, phoneme_code[val][2] )
	write ( filename, 11000, output )
	#print ( output )


#================================================
#test cases


#doors; Hello.  This was separately text-to-speeched.
testcase_001 = [
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Won't you tell me your name?"
0x2E, 0x35, 0x0B, 0x0D, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x0D, 0x07, 0x2D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x0B, 0x14, 0x36, 0x10, 0x03, 0x02, 0x04, 0x04, 0x03, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Let me jump in your game"
0x2D, 0x07, 0x0D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x0A, 0x0F, 0x10, 0x09, 0x03, 0x02, 0x0C, 0x0C, 0x0B, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x24, 0x14, 0x36, 0x10, 0x03, 0x02, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Won't you tell me your name?"
0x2E, 0x35, 0x0B, 0x0D, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x0D, 0x07, 0x2D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x0B, 0x14, 0x36, 0x10, 0x03, 0x02, 0x04, 0x04, 0x03, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Let me jump in your game"
0x2D, 0x07, 0x0D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x0A, 0x0F, 0x10, 0x09, 0x03, 0x02, 0x0C, 0x0C, 0x0B, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x24, 0x14, 0x36, 0x10, 0x03, 0x02, 

#"She's walking down the street"
0x25, 0x07, 0x2B, 0x03, 0x02, 0x2E, 0x17, 0x17, 0x2A, 0x0C, 0x2C, 0x03, 0x02, 0x21, 0x20, 0x0B, 0x03, 0x02, 0x12, 0x13, 0x03, 0x02, 0x37, 0x0D, 0x27, 0x13, 0x0D, 0x03, 0x02, 
#"Blind to every eye she meets"
0x1C, 0x2D, 0x06, 0x0B, 0x15, 0x03, 0x02, 0x0D, 0x1F, 0x03, 0x02, 0x07, 0x07, 0x23, 0x1B, 0x34, 0x13, 0x03, 0x02, 0x18, 0x06, 0x03, 0x02, 0x25, 0x13, 0x03, 0x02, 0x10, 0x13, 0x11, 0x37, 0x03, 0x02, 
#"Do you think you'll be the guy"
0x21, 0x1F, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x1D, 0x0C, 0x2C, 0x2A, 0x03, 0x02, 0x19, 0x1F, 0x00, 0x2D, 0x03, 0x02, 0x3F, 0x13, 0x03, 0x02, 0x12, 0x13, 0x03, 0x02, 0x3D, 0x18, 0x06, 0x03, 0x02, 
#"To make the queen of the angels sigh?"
0x0D, 0x1F, 0x03, 0x02, 0x10, 0x14, 0x36, 0x2A, 0x03, 0x02, 0x12, 0x13, 0x03, 0x02, 0x08, 0x2E, 0x13, 0x0B, 0x03, 0x02, 0x0F, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x12, 0x13, 0x03, 0x02, 0x14, 0x36, 0x0B, 0x0A, 0x07, 0x2D, 0x2B, 0x03, 0x02, 0x37, 0x06, 0x03, 0x02, 0x04, 0x04, 0x03, 

#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Won't you tell me your name?"
0x2E, 0x35, 0x0B, 0x0D, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x0D, 0x07, 0x2D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x0B, 0x14, 0x36, 0x10, 0x03, 0x02, 0x04, 0x04, 0x03, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Let me jump in your game"
0x2D, 0x07, 0x0D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x0A, 0x0F, 0x10, 0x09, 0x03, 0x02, 0x0C, 0x0C, 0x0B, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x24, 0x14, 0x36, 0x10, 0x03, 0x02, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Won't you tell me your name?"
0x2E, 0x35, 0x0B, 0x0D, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x0D, 0x07, 0x2D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x0B, 0x14, 0x36, 0x10, 0x03, 0x02, 0x04, 0x04, 0x03, 
#"Hello, I love you"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x17, 0x06, 0x03, 0x02, 0x2D, 0x0F, 0x23, 0x1B, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Let me jump in your game"
0x2D, 0x07, 0x0D, 0x03, 0x02, 0x10, 0x13, 0x03, 0x02, 0x0A, 0x0F, 0x10, 0x09, 0x03, 0x02, 0x0C, 0x0C, 0x0B, 0x03, 0x02, 0x19, 0x1E, 0x34, 0x03, 0x02, 0x24, 0x14, 0x36, 0x10, 0x03, 0x02, 

#"She holds her head so high"
0x25, 0x13, 0x03, 0x02, 0x1B, 0x35, 0x2D, 0x21, 0x2B, 0x03, 0x02, 0x1B, 0x34, 0x03, 0x02, 0x1B, 0x13, 0x21, 0x03, 0x02, 0x37, 0x35, 0x03, 0x02, 0x1B, 0x06, 0x03, 0x02, 
#"Like a statue in the sky"
0x2D, 0x06, 0x2A, 0x03, 0x02, 0x07, 0x14, 0x36, 0x03, 0x02, 0x37, 0x0D, 0x1A, 0x0D, 0x1F, 0x03, 0x02, 0x0C, 0x0C, 0x0B, 0x03, 0x02, 0x12, 0x13, 0x03, 0x02, 0x37, 0x2A, 0x13, 0x03, 0x02, 
#"Her arms are wicked, and her legs are long"
0x1B, 0x34, 0x03, 0x02, 0x1A, 0x33, 0x10, 0x2B, 0x03, 0x02, 0x18, 0x34, 0x03, 0x02, 0x2E, 0x0C, 0x29, 0x15, 0x03, 0x02, 0x04, 0x1A, 0x0B, 0x15, 0x03, 0x02, 0x1B, 0x34, 0x03, 0x02, 0x2D, 0x07, 0x22, 0x2B, 0x03, 0x02, 0x18, 0x34, 0x03, 0x02, 0x2D, 0x17, 0x2C, 0x03, 0x02, 
#"When she moves my brain screams out this song"
0x30, 0x07, 0x0B, 0x03, 0x02, 0x25, 0x13, 0x03, 0x02, 0x10, 0x1F, 0x23, 0x1B, 0x2B, 0x03, 0x02, 0x10, 0x06, 0x03, 0x02, 0x1C, 0x27, 0x07, 0x14, 0x36, 0x0B, 0x03, 0x02, 0x37, 0x08, 0x27, 0x13, 0x10, 0x2B, 0x03, 0x02, 0x20, 0x0D, 0x03, 0x02, 0x36, 0x0C, 0x0C, 0x37, 0x37, 0x03, 0x02, 0x37, 0x17, 0x2C, 0x03, 0x02, 

#"Sidewalk crouches at her feet"
0x37, 0x0C, 0x21, 0x2E, 0x17, 0x17, 0x2A, 0x03, 0x02, 0x08, 0x27, 0x20, 0x32, 0x0C, 0x2B, 0x03, 0x02, 0x1A, 0x0D, 0x03, 0x02, 0x1B, 0x34, 0x03, 0x02, 0x28, 0x13, 0x0D, 0x03, 0x02, 
#"Like a dog that begs for something sweet"
0x2D, 0x06, 0x2A, 0x03, 0x02, 0x07, 0x14, 0x36, 0x03, 0x02, 0x21, 0x17, 0x22, 0x03, 0x02, 0x36, 0x1A, 0x0D, 0x03, 0x02, 0x3F, 0x07, 0x22, 0x2B, 0x03, 0x02, 0x28, 0x17, 0x17, 0x33, 0x03, 0x02, 0x37, 0x0F, 0x10, 0x1D, 0x0C, 0x2C, 0x03, 0x02, 0x37, 0x2E, 0x13, 0x0D, 0x03, 0x02, 
#"Do you hope to make her see, you fool?"
0x21, 0x1F, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x1B, 0x35, 0x09, 0x03, 0x02, 0x0D, 0x1F, 0x03, 0x02, 0x10, 0x14, 0x36, 0x2A, 0x03, 0x02, 0x1B, 0x34, 0x03, 0x02, 0x37, 0x13, 0x03, 0x02, 0x04, 0x19, 0x1F, 0x03, 0x02, 0x28, 0x1F, 0x2D, 0x03, 0x02, 0x04, 0x04, 0x03, 
#"Do you hope to pluck this dusky jewel?"
0x21, 0x1F, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 0x1B, 0x35, 0x09, 0x03, 0x02, 0x0D, 0x1F, 0x03, 0x02, 0x09, 0x2D, 0x0F, 0x29, 0x03, 0x02, 0x36, 0x0C, 0x0C, 0x37, 0x37, 0x03, 0x02, 0x21, 0x0F, 0x37, 0x2A, 0x13, 0x03, 0x02, 0x0A, 0x1F, 0x07, 0x2D, 0x03, 0x02, 0x04, 0x04, 0x03, 

#"Hello, Hello, Hello, Hello, Hello, Hello, Hello"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 
#"I want you"
0x17, 0x06, 0x03, 0x02, 0x2E, 0x18, 0x18, 0x0B, 0x0D, 0x03, 0x02, 0x19, 0x1F, 0x03, 0x02, 
#"Hello"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 
#"I need my baby"
0x17, 0x06, 0x03, 0x02, 0x0B, 0x13, 0x15, 0x03, 0x02, 0x10, 0x06, 0x03, 0x02, 0x3F, 0x14, 0x36, 0x3F, 0x13, 0x03, 0x02, 
#"Hello, Hello, Hello, Hello"
0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 0x04, 0x1B, 0x07, 0x2D, 0x35, 0x03, 0x02, 

]


#whizanddump ( 'testcase_001.wav', testcase_001 )




#=========================================================================
#Text2Speech rules
#XXX WIP as I try to understand my original work

#This implementation is derived from the following research:
'''
AUTOMATIC TRANSLATION OF ENGLISH TEXT TO PHONETICS
BY MEANS OF LETTER-TO-SOUND RULES

NRL Report 7948

January 21st, 1976
Naval Research Laboratory, Washington, D.C.

Published by the National Technical Information Service as
document "AD/A021 929".
'''

#additionally, this implementation is derived from a work by
#John A. Wasser which the author placed into the public domain.

#additionally, this implementation uses additional rules
#presumably developed by Tom Jennings for his t2a program.

#additionally, I (ziggurat29) added a couple mods of my own here and there

#'rules' are a tuple of ( "left context", "bracket context", "right context", "phoneme list" )
#the way rules work are that the prefix, bracket, suffix must match literally.  If they
#do, then the phoneme list is emitted.

#I called the middle part that is being matched and replaced, the 'bracket' context,
#because in the original text the rules are written as:
#   a[b]c=d



#the NRL Report 7948 describes a system containing a small-ish set of rules
#that match text patterns, replacing them with phoneme sequences.  As
#described, the rules work by attempting to match a text sequence (to be
#replaced) with the match being subject to the 'context' of the surrounding
#characters (which contribute to the match, but are not themselves
#replaced upon match).

#the left and right context matches have some enhancements:  literal match for
#alphabetic characters, and the apostrophe, and space, and some meta
#character classes represented by these symbols:
#  #  one or more vowels
#  :  zero or more consonants
#  ^  one consonant
#  .  one voiced consonant
#  %  'e'-related things at the end of the word '-e', '-ed', '-er', '-es', '-ely', '-ing'
#  +  'front' vowels 'e', 'i', 'y'

#note:  I originally intended to use regexes instead of this bespoke
#implementation, but there were too many rules to hold all the machines,
#and anyway that would be less easily to port to MCUs.


#To expedite the matching process, the rules are grouped according to the first
#character in the left context.  This tweak avoids testing most of the rules
#that have no chance of matching.

#A group of rules is processed linearly, so more specific rules should precede
#more general ones; the last rule should be a catchall for the group.

#The empty string represents 'anything', and the single space string represents
#beginning or end of line' (this works because text is processed word-by-word,
#and is given a leading and trailing space).


#(syntatic sugar for readability)
Silent = []
Anything = ""
Nothing = " "

#symbolic 'constants' for readability
PA1 = 0x00	#PAUSE		 1OMS
PA2 = 0x01	#PAUSE		 30MS
PA3 = 0x02	#PAUSE		 50MS
PA4 = 0x03	#PAUSE		1OOMS
PA5 = 0x04	#PAUSE		200MS
OY  = 0x05	#bOY		420MS
AY  = 0x06	#skY		260MS
EH  = 0x07	#End		 70MS
KK3 = 0x08	#Comb		120MS
PP  = 0x09	#Pow		21OMS
JH  = 0x0a	#doDGe		140MS
NN1 = 0x0b	#thiN		140MS
IH  = 0x0c	#sIt		 70MS
TT2 = 0x0d	#To			140MS
RR1 = 0x0e	#Rural		170MS
AX  = 0x0f	#sUcceed	 70MS
AH  = 0x0f	#(pseudo-phoneme)	XXX have to fixup rules; scrutinize this
MM  = 0x10	#Milk		180MS
TT1 = 0x11	#parT		1OOMS
DH1 = 0x12	#THey		290MS
IY  = 0x13	#sEE		250MS
EY  = 0x14	#bEIge		280MS
DD1 = 0x15	#coulD		 70MS
UW1 = 0x16	#tO			1OOMS
AO  = 0x17	#AUght		1OOMS
AA  = 0x18	#hOt		1OOMS
YY2 = 0x19	#Yes		180MS
AE  = 0x1a	#hAt		120MS
HH1 = 0x1b	#He			130MS
BB1 = 0x1c	#Business	 80MS
TH  = 0x1d	#THin		180MS
UH  = 0x1e	#bOOk		100MS
UW2 = 0x1f	#fOOd		260MS
AW  = 0x20	#OUt		370MS
DD2 = 0x21	#Do			160MS
GG3 = 0x22	#wiG		140MS
VV  = 0x23	#Vest		19OMS
GG1 = 0x24	#Got		 80MS
SH  = 0x25	#SHip		160MS
ZH  = 0x26	#aZure		190MS
RR2 = 0x27	#bRain		12OMS
FF  = 0x28	#Food		150MS
KK2 = 0x29	#sKy		190MS
KK1 = 0x2a	#Can't		160MS
ZZ  = 0x2b	#Zoo		21OMS
NG  = 0x2c	#aNchor		220MS
LL  = 0x2d	#Lake		110MS
WW  = 0x2e	#Wool		180MS
XR  = 0x2f	#repAIR		360MS
WH  = 0x30	#WHig		200MS
YY1 = 0x31	#Yes		130MS
CH  = 0x32	#CHurch		190MS
ER1 = 0x33	#fIR		160MS
ER2 = 0x34	#fIR		300MS
OW  = 0x35	#bEAU		240MS
DH2 = 0x36	#THey		240MS
SS  = 0x37	#veSt		 90MS
NN2 = 0x38	#No			190MS
HH2 = 0x39	#Hoe		180MS
OR  = 0x3a	#stORe		330MS
AR  = 0x3b	#alARm		290MS
YR  = 0x3c	#clEAR		350MS
GG2 = 0x3d	#Guest		 40MS
EL  = 0x3e	#saddLe		190MS
BB2 = 0x3f	#Business	 50MS



#0 - punctuation
r_punc = [
	[ Anything,		" ",		Anything,	[ PA4, PA3, ]	],
	[ Anything,		"-",		Anything,	[ PA4, ] 	],
	[ ".",			"'s",		Anything,	[ ZZ, ] 	],
	[ "#:.e",		"'s",		Anything,	[ ZZ, ] 	],
	[ "#",			"'s",		Anything,	[ ZZ, ] 	],
	[ Anything,		"'",		Anything,	[ PA1, ] 	],
	[ Anything,		";",		Anything,	[ PA5, ] 	],
	[ Anything,		":",		Anything,	[ PA5, ] 	],
	[ Anything,		",",		Anything,	[ PA5, ] 	],

	[ Anything,		".",		"#",		Silent	],
	[ Anything,		".",		"^",		Silent	],
	[ Anything,		".",		Anything,	[ PA5, PA5, PA4, ] 	],

	[ Anything,		"?",		Anything,	[ PA5, PA5, PA4, ] 	],
	[ Anything,		"!",		Anything,	[  PA5, PA5, PA4, ] 	],
]


#1 - a
r_a = [
	[ Nothing,		"a",		Nothing,	[ EH, EY, ] 	],
	[ Anything,		"ahead",	Anything,	[ AX, HH1, EH, EH, DD1, ] 	],
	[ Anything,		"apropos",	Anything,	[ AE, PP, ER1, OW, PP, OW, ] 	],
	[ Anything,		"ass",		"h",		[ AE, AE, SS, SS, ] 	],
	[ Anything,		"allege",	Anything,	[ AX, LL, EH, DD2, JH, ] 	],
	[ Anything,		"again",	Anything,	[ AX, GG3, EH, EH, NN1, ] 	],
	[ Nothing,		"able",		Anything,	[ EY, HH1, BB2, AX, LL, ] 	],
	[ Nothing,		"above",	Nothing,	[ AX, BB2, AX, AX, VV, HH1, ] 	],
	[ Nothing,		"acro",		".",		[ AE, HH1, KK1, ER1, OW, ] 	],
	[ Nothing,		"are",		Nothing,	[ AA, ER2, ] 	],
	[ Nothing,		"ally",		Nothing,	[ AE, AE, LL, AY, ] 	],
	[ Anything,		"atomic",	Anything,	[ AX, TT2, AA, MM, PA1, IH, KK1, ] 	],
	[ Anything,		"arch",		"#v",		[ AX, AX, ER1, PA1, KK1, IH, ] 	],
	[ Anything,		"arch",		"#.",		[ AX, AX, ER1, CH, IH, ] 	],
	[ Anything,		"arch",		"#^",		[ AX, AX, ER1, KK1, PA1, IH, ] 	],
	[ Anything,		"argue",	Anything,	[ AA, ER2, GG1, YY2, UW2, ] 	],

	[ Nothing,		"abb",		Anything,	[ AX, AX, BB2, ] 	],
	[ Nothing,		"ab",		Anything,	[ AE, AE, BB1, PA2, ] 	],
	[ Nothing,		"an",		"#",		[ AE, NN1, ] 	],
	[ Nothing,		"allo",		"t",		[ AE, LL, AA, ] 	],
	[ Nothing,		"allo",		"w",		[ AE, LL, AW, ] 	],
	[ Nothing,		"allo",		Anything,	[ AE, LL, OW, ] 	],
	[ Nothing,		"ar",		"o",		[ AX, ER2, ] 	],

	[ "#:",			"ally",		Anything,	[ PA1, AX, LL, IY, ] 	],
	[ "^",			"able",		Anything,	[ PA1, EY, HH1, BB2, AX, LL, ] 	],
	[ Anything,		"able",		Anything,	[ PA1, AX, HH1, BB2, AX, LL, ] 	],
	[ "^",			"ance",		Anything,	[ PA1, AE, NN1, SS, ] 	],
	[ Anything,		"air",		Anything,	[ EY, XR, ] 	],
	[ Anything,		"aic",		Nothing,	[ EY, IH, KK1, ] 	],
	[ "#:",			"als",		Nothing,	[ AX, LL, ZZ, ] 	],
	[ Anything,		"alk",		Anything,	[ AO, AO, KK1, ] 	],
	[ Anything,		"arr",		Anything,	[ AA, ER1, ] 	],
	[ Anything,		"ang",		"+",		[ EY, NN1, JH, ] 	],
	[ " :",			"any",		Anything,	[ EH, NN1, IY, ] 	],
	[ Anything,		"ary",		Nothing,	[ PA1, AX, ER2, IY, ] 	],
	[ "^",			"as",		"#",		[ EY, SS, ] 	],
	[ "#:",			"al",		Nothing,	[ AX, LL, ] 	],
	[ Anything,		"al",		"^",		[ AO, LL, ] 	],
	[ Nothing,		"al",		"#",		[ EH, EY, LL, ] 	],
	[ "#:",			"ag",		"e",		[ IH, JH, ] 	],

	[ Anything,		"ai",		Anything,	[ EH, EY, ] 	],
	[ Anything,		"ay",		Anything,	[ EH, EY, ] 	],
	[ Anything,		"au",		Anything,	[ AO, AO, ] 	],
	[ Anything,		"aw",		Nothing,	[ AO, AO, ] 	],
	[ Anything,		"aw",		"^",		[ AO, AO, ] 	],
	[ ":",			"ae",		Anything,	[ EH, ] 	],
	[ Anything,		"a",		"tion",		[ EY, ] 	],
	[ "c",			"a",		"bl",		[ EH, EY, ] 	],
	[ "c",			"a",		"b#",		[ AE, AE, ] 	],
	[ "c",			"a",		"pab",		[ EH, EY, ] 	],
	[ "c",			"a",		"p#",		[ AE, AE, ] 	],
	[ "c",			"a",		"t#^",		[ AE, AE, ] 	],

	[ "^^^",		"a",		Anything,	[ EY, ] 	],
	[ "^.",			"a",		"^e",		[ EY, ] 	],
	[ "^.",			"a",		"^i",		[ EY, ] 	],
	[ "^^",			"a",		Anything,	[ AE, ] 	],
	[ "^",			"a",		"^##",		[ EY, ] 	],
	[ "^",			"a",		"^#",		[ EY, ] 	],
	[ "^",			"a",		"^#",		[ EH, EY, ] 	],
	[ Anything,		"a",		"^%",		[ EY, ] 	],
	[ "#",			"a",		Nothing,	[ AO, ] 	],
	[ Anything,		"a",		"wa",		[ AX, ] 	],
	[ Anything,		"a",		Nothing,	[ AX, ] 	],
	[ Anything,		"a",		"^+#",		[ EY, ] 	],
	[ Anything,		"a",		"^+:#",		[ AE, ] 	],
	[ " :",			"a",		"^+ ",		[ EY, ] 	],

	[ Anything,		"a",		Anything,	[ AE, ] 	],
]

#2 - b
r_b = [
	[ "b",			"b",		Anything,	Silent	],
	[ Anything,		"bi",		"cycle",	[ BB2, AY, ] 	],
	[ Anything,		"bi",		"cycle",	[ BB2, AY, ] 	],
	[ Anything,		"bbq",		Anything,	[ BB2, AX, AX, ER1, BB2, AX, KK2, YY2, UW2, ] 	],
	[ Anything,		"barbeque",	Anything,	[ BB2, AX, AX, ER1, BB2, AX, KK2, YY2, UW2, ] 	],
	[ Anything,		"barbaque",	Anything,	[ BB2, AX, AX, ER1, BB2, AX, KK2, YY2, UW2, ] 	],
	[ Anything,		"bargain",	Anything,	[ BB2, AO, ER1, GG1, EH, NN1, ] 	],
	[ Anything,		"bagel",	Anything,	[ BB2, EY, GG1, EH, LL, ] 	],
	[ Anything,		"being",	Anything,	[ BB2, IY, IH, NG, ] 	],
	[ Anything,		"bomb",		Anything,	[ BB2, AA, AA, MM, ] 	],
	[ Nothing,		"both",		Nothing,	[ BB2, OW, TH, ] 	],
	[ Anything,		"buil",		Anything,	[ BB2, IH, LL, ] 	],
	[ Nothing,		"bus",		"y",		[ BB2, IH, ZZ, ] 	],
	[ Nothing,		"bus",		"#",		[ BB2, IH, ZZ, ] 	],
	[ Anything,		"bye",		Anything,	[ BB2, AO, AY, ] 	],
	[ Anything,		"bear",		Nothing,	[ BB2, EY, ER2, ] 	],
	[ Anything,		"bear",		"%",		[ BB2, EY, ER2, ] 	],
	[ Anything,		"bear",		"s",		[ BB2, EY, ER2, ] 	],
	[ Anything,		"bear",		"#",		[ BB2, EY, ER2, ] 	],
	[ Nothing,		"beau",		Anything,	[ BB2, OW, ] 	],

	[ Anything,		"ban",		"ish",		[ BB2, AE, AE, NN1, ] 	],

	[ Nothing,		"be",		"^#",		[ BB2, IH, ] 	],
	[ Nothing,		"by",		Anything,	[ BB2, AO, AY, ] 	],
	[ "y",			"be",		Nothing,	[ BB2, IY, ] 	],

	[ Nothing,		"b",		"#",		[ BB2, ] 	],
	[ Anything,		"b",		Nothing,	[ BB1, ] 	],
	[ Anything,		"b",		"#",		[ BB1, ] 	],
	[ Anything,		"b",		"l",		[ BB1, ] 	],
	[ Anything,		"b",		"r",		[ BB1, ] 	],

	[ Anything,		"b",		Anything,	[ BB2, ] 	],
]

#3 - c
r_c = [
	[ Anything,		"chinese",	Anything,	[ CH, AY, NN1, IY, SS, ] 	],
	[ Anything,		"country",	Anything,	[ KK1, AX, AX, NN1, TT2, ER1, IY, ] 	],
	[ Anything,		"christ",	Nothing,	[ KK3, ER1, AY, SS, TT2, ] 	],
	[ Anything,		"chassis",	Anything,	[ CH, AX, AX, SS, IY, ] 	],
	[ Anything,		"closet",	Anything,	[ KK3, LL, AO, AO, ZZ, EH, TT2, ] 	],
	[ Anything,		"china",	Anything,	[ CH, AY, NN1, AX, ] 	],
	[ Nothing,		"cafe",		Nothing,	[ KK1, AE, FF, AE, EY, ] 	],
	[ Anything,		"cele",		Anything,	[ SS, EH, LL, PA1, EH, ] 	],
	[ Anything,		"cycle",	Anything,	[ SS, AY, KK3, UH, LL, ] 	],
	[ Anything,		"chron",	Anything,	[ KK1, ER1, AO, NN1, ] 	],
	[ Anything,		"crea",		"t",		[ KK3, ER1, IY, EY, ] 	],
	[ Nothing,		"cry",		Nothing,	[ KK3, ER1, IY, ] 	],
	[ Nothing,		"chry",		Anything,	[ KK3, ER1, AO, AY, ] 	],
	[ Nothing,		"cry",		"#",		[ KK3, ER1, AO, AY, ] 	],
	[ Nothing,		"caveat",	":",		[ KK1, AE, VV, IY, AE, TT2, ] 	],
	[ "^",			"cuit",		Anything,	[ KK1, IH, TT2, ] 	],
	[ Anything,		"chaic",	Anything,	[ KK1, EY, IH, KK1, ] 	],
	[ Anything,		"cation",	Anything,	[ KK1, EY, SH, AX, NN1, ] 	],
	[ Nothing,		"ch",		"aract",	[ KK1, ] 	],
	[ Nothing,		"ch",		"^",		[ KK1, ] 	],
	[ "^e",			"ch",		Anything,	[ KK1, ] 	],
	[ Anything,		"ch",		Anything,	[ CH, ] 	],
	[ " s",			"ci",		"#",		[ SS, AY, ] 	],
	[ Anything,		"ci",		"a",		[ SH, ] 	],
	[ Anything,		"ci",		"o",		[ SH, ] 	],
	[ Anything,		"ci",		"en",		[ SH, ] 	],
	[ Anything,		"c",		"+",		[ SS, ] 	],
	[ Anything,		"ck",		Anything,	[ KK2, ] 	],
	[ Anything,		"com",		"%",		[ KK1, AH, MM, ] 	],
	#[ Anything,	"c",		"^",		[ KK3, ] 	],

	[ Anything,		"c",		"u",		[ KK3, ] 	],
	[ Anything,		"c",		"o",		[ KK3, ] 	],
	[ Anything,		"c",		"a^^",		[ KK3, ] 	],
	[ Anything,		"c",		"o^^",		[ KK3, ] 	],
	[ Anything,		"c",		"l",		[ KK3, ] 	],
	[ Anything,		"c",		"r",		[ KK3, ] 	],

	[ Anything,		"c",		"a",		[ KK1, ] 	],
	[ Anything,		"c",		"e",		[ KK1, ] 	],
	[ Anything,		"c",		"i",		[ KK1, ] 	],

	[ Anything,		"c",		Nothing,	[ KK2, ] 	],
	[ Anything,		"c",		Anything,	[ KK1, ] 	],
]

#d 10
r_d = [
	[ Anything,		"dead",		Anything,	[ DD2, EH, EH, DD1, ] 	],
	[ Nothing,		"dogged",	Anything,	[ DD2, AO, GG1, PA1, EH, DD1, ] 	],
	[ "#:",			"ded",		Nothing,	[ DD2, IH, DD1, ] 	],
	[ Nothing,		"dig",		Anything,	[ DD2, IH, IH, GG1, ] 	],
	[ Nothing,		"dry",		Nothing,	[ DD2, ER1, AO, AY, ] 	],
	[ Nothing,		"dry",		"#",		[ DD2, ER1, AO, AY, ] 	],
	[ Nothing,		"de",		"^#",		[ DD2, IH, ] 	],
	[ Nothing,		"do",		Nothing,	[ DD2, UW2, ] 	],
	[ Nothing,		"does",		Anything,	[ DD2, AH, ZZ, ] 	],
	[ Nothing,		"doing",	Anything,	[ UW2, IH, NG, ] 	],
	[ Nothing,		"dow",		Anything,	[ DD2, AW, ] 	],
	[ Anything,		"du",		"a",		[ JH, UW2, ] 	],
	[ Anything,		"dyna",		Anything,	[ DD2, AY, NN1, AX, PA1, ] 	],
	[ Anything,		"dyn",		"#",		[ DD2, AY, NN1, PA1, ] 	],
	[ "d",			"d",		Anything,	Silent	],
	[ Anything,		"d",		Nothing,	[ DD1, ] 	],
	[ Nothing,		"d",		Anything,	[ DD2, ] 	],
	[ Anything,		"d",		Anything,	[ DD2, ] 	],
]

#e 52
r_e = [
	[ Nothing,		"eye",		Anything,	[ AA, AY, ] 	],
	[ Anything,		"ered",		Nothing,	[ ER2, DD1, ] 	],
	[ Nothing,		"ego",		Anything,	[ IY, GG1, OW, ] 	],
	[ Nothing,		"err",		Anything,	[ EH, EH, ER1, ] 	],
	[ "^",			"err",		Anything,	[ EH, EH, ER1, ] 	],
	[ Anything,		"ev",		"er",		[ EH, EH, VV, HH1, ] 	],
	[ Anything,		"e",		"ness",		Silent	],
	#[ Anything,		"e",		"^%",		"IY, ] 	],
	[ Anything,		"eri",		"#",		[ IY, XR, IY, ] 	],
	[ Anything,		"eri",		Anything,	[ EH, ER1, IH, ] 	],
	[ "#:",			"er",		"#",		[ ER2, ] 	],
	[ Anything,		"er",		"#",		[ EH, EH, ER1, ] 	],
	[ Anything,		"er",		Anything,	[ ER2, ] 	],
	[ Nothing,		"evil",		Anything,	[ IY, VV, EH, LL, ] 	],
	[ Nothing,		"even",		Anything,	[ IY, VV, EH, NN1, ] 	],
	[ "m",			"edia",		Anything,	[ IY, DD2, IY, AX, ] 	],
	[ Anything,		"ecia",		Anything,	[ IY, SH, IY, EY, ] 	],
	[ ":",			"eleg",		Anything,	[ EH, LL, EH, GG1, ] 	],

	[ "#:",			"e",		"w",		Silent	],
	[ "t",			"ew",		Anything,	[ UW2, ] 	],
	[ "s",			"ew",		Anything,	[ UW2, ] 	],
	[ "r",			"ew",		Anything,	[ UW2, ] 	],
	[ "d",			"ew",		Anything,	[ UW2, ] 	],
	[ "l",			"ew",		Anything,	[ UW2, ] 	],
	[ "z",			"ew",		Anything,	[ UW2, ] 	],
	[ "n",			"ew",		Anything,	[ UW2, ] 	],
	[ "j",			"ew",		Anything,	[ UW2, ] 	],
	[ "th",			"ew",		Anything,	[ UW2, ] 	],
	[ "ch",			"ew",		Anything,	[ UW2, ] 	],
	[ "sh",			"ew",		Anything,	[ UW2, ] 	],
	[ Anything,		"ew",		Anything,	[ YY2, UW2, ] 	],
	[ Anything,		"e",		"o",		[ IY, ] 	],
	[ "#:s",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:c",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:g",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:z",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:x",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:j",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:ch",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:sh",		"es",		Nothing,	[ IH, ZZ, ] 	],
	[ "#:",			"e",		"s ",		Silent	],
	[ "#:",			"ely",		Nothing,	[ LL, IY, ] 	],
	[ "#:",			"ement",	Anything,	[ PA1, MM, EH, NN1, TT2, ] 	],
	[ Anything,		"eful",		Anything,	[ PA1, FF, UH, LL, ] 	],
	[ Anything,		"ee",		Anything,	[ IY, ] 	],
	[ Anything,		"earn",		Anything,	[ ER2, NN1, ] 	],
	[ Nothing,		"ear",		"^",		[ ER2, ] 	],
	[ "k.",			"ead",		Anything,	[ IY, DD2, ] 	],
	[ "^.",			"ead",		Anything,	[ EH, DD2, ] 	],
	[ "d",			"ead",		Anything,	[ EH, DD2, ] 	],
	[ Anything,		"ead",		Anything,	[ IY, DD2, ] 	],
	[ "#:",			"ea",		Nothing,	[ IY, AX, ] 	],
	[ "#:",			"ea",		"s",		[ IY, AX, ] 	],
	[ Anything,		"ea",		"su",		[ EH, ] 	],
	[ Anything,		"ea",		Anything,	[ IY, ] 	],
	[ Anything,		"eigh",		Anything,	[ EY, ] 	],
	[ "l",			"ei",		Anything,	[ IY, ] 	],
	[ ".",			"ei",		Anything,	[ EY, ] 	],
	[ Anything,		"ei",		"n",		[ AY, ] 	],
	[ Anything,		"ei",		Anything,	[ IY, ] 	],
	[ Anything,		"ey",		Anything,	[ IY, ] 	],
	[ Anything,		"eu",		Anything,	[ YY2, UW2, ] 	],

	[ "#:",			"e",		"d ",		Silent	],
	[ "#s",			"e",		"^",		Silent	],
	[ ":",			"e",		"x",		[ EH, EH, ] 	],
	[ "#:",			"e",		Nothing,	Silent	],
	[ "+:",			"e",		Nothing,	Silent	],
	[ "':^",		"e",		Nothing,	Silent	],
	[ ":",			"equ",		Anything,	[ IY, KK1, WW, ] 	],
	[ "dg",			"e",		Anything,	Silent	],
	[ "dh",			"e",		Anything,	[ IY, ] 	],
	[ " :",			"e",		Nothing,	[ IY, ] 	],
	[ "#",			"ed",		Nothing,	[ DD1, ] 	],
	[ Anything,		"e",		Anything,	[ EH, ] 	],
]

#f 2
r_f = [
	[ Anything,		"fnord",	Anything,	[ FF, NN1, AO, OR, DD1, ] 	],
	[ Anything,		"four",		Anything,	[ FF, OW, ER1, ] 	],
	[ Anything,		"ful",		Anything,	[ PA1, FF, UH, LL, ] 	],
	[ Nothing,		"fly",		Anything,	[ FF, LL, AO, AY, ] 	],
	[ ".",			"fly",		Anything,	[ FF, LL, AO, AY, ] 	],
	[ Anything,		"fixed",	Anything,	[ FF, IH, KK1, SS, TT2, ] 	],
	[ Anything,		"five",		Anything,	[ FF, AO, AY, VV, ] 	],
	[ Anything,		"foot",		Anything,	[ FF, UH, UH, TT2, ] 	],
	[ Anything,		"f",		Anything,	[ FF, ] 	],
]

#g 10
r_g = [
	[ Anything,		"gadget",	Anything,	[ GG2, AE, AE, DD1, PA2, JH, EH, EH, TT2, ] 	],
	[ Anything,		"god",		Anything,	[ GG3, AA, AA, DD1, ] 	],
	[ Anything,		"get",		Anything,	[ GG3, EH, EH, TT2, ] 	],
	[ Anything,		"gen",		"^",		[ JH, EH, EH, NN1, ] 	],
	[ Anything,		"gen",		"#^",		[ JH, EH, EH, NN1, ] 	],
	[ Anything,		"gen",		Nothing,	[ JH, EH, EH, NN1, ] 	],
	[ Anything,		"giv",		Anything,	[ GG2, IH, IH, VV, HH1, ] 	],
	[ "su",			"gges",		Anything,	[ GG1, JH, EH, SS, ] 	],
	[ Anything,		"great",	Anything,	[ GG2, ER1, EY, TT2, ] 	],
	[ Anything,		"good",		Anything,	[ GG2, UH, UH, DD1, ] 	],
	#hmmm guest guess
	[ Nothing,		"gue",		Anything,	[ GG2, EH, ] 	],
	#hmm don't know about this one.  argue? vague?
	[ Anything,		"gue",		Anything,	[ GG3, ] 	],

	[ "d",			"g",		Anything,	[ JH, ] 	],
	[ "##",			"g",		Anything,	[ GG1, ] 	],
	[ Anything,		"g",		"+",		[ JH, ] 	],
	[ Anything,		"gg",		Anything,	[ GG3, PA1, ] 	],

	[ "campai",		"g",		"n",		Silent	],
	[ "arrai",		"g",		"n",		Silent	],
	[ "ali",		"g",		"n",		Silent	],
	[ "beni",		"g",		"n",		Silent	],
	[ "arrai",		"g",		"n",		Silent	],

	[ Anything,		"g",		"a",		[ GG1, ] 	],
	[ Anything,		"g",		"e",		[ GG1, ] 	],
	[ Anything,		"g",		"i",		[ GG1, ] 	],
	[ Anything,		"g",		"y",		[ GG1, ] 	],

	[ Anything,		"g",		"o",		[ GG2, ] 	],
	[ Anything,		"g",		"u",		[ GG2, ] 	],
	[ Anything,		"g",		"l",		[ GG2, ] 	],
	[ Anything,		"g",		"r",		[ GG2, ] 	],


	[ Anything,		"g",		Nothing,	[ GG3, ] 	],
	[ "n",			"g",		Anything,	[ GG3, ] 	],
	[ Anything,		"g",		Anything,	[ GG3, ] 	],
]

#h 6
r_h = [
	[ Anything,		"honor",	Anything,	[ AO, NN1, ER2, ] 	],
	[ Anything,		"heard",	Anything,	[ HH1, ER2, DD1, ] 	],
	[ Anything,		"height",	Anything,	[ HH1, AY, TT2, ] 	],
	[ Anything,		"honest",	Anything,	[ AO, NN1, EH, SS, TT2, ] 	],
	[ Anything,		"hood",		Anything,	[ HH1, UH, UH, DD1, ] 	],
	[ "ab",			"hor",		Anything,	[ OW, ER2, ] 	],
	[ Anything,		"heavy",	Anything,	[ HH1, AE, VV, IY, ] 	],
	[ Anything,		"heart",	Anything,	[ HH1, AA, ER1, TT2, ] 	],
	[ Anything,		"half",		Anything,	[ HH1, AE, AE, FF, ] 	],
	[ Anything,		"hive",		Anything,	[ HH1, AA, AY, VV, ] 	],
	[ Anything,		"heavi",	":#",		[ HH1, AE, VV, IY, ] 	],
	[ Nothing,		"hav",		Anything,	[ HH1, AE, VV, HH1, ] 	],
	[ Anything,		"ha",		Nothing,	[ HH1, AA, AA, ] 	],
	[ Nothing,		"hi",		Nothing,	[ HH1, AA, AY, ] 	],
	[ Anything,		"he",		"t",		[ HH1, AE, ] 	],
	[ Anything,		"he",		"x",		[ HH1, AE, ] 	],
	[ Anything,		"hy",		Anything,	[ HH1, AA, AY, ] 	],
	[ Nothing,		"hang",		Anything,	[ HH1, AE, NG, ] 	],
	[ Nothing,		"here",		Anything,	[ HH1, IY, XR, ] 	],
	[ Nothing,		"hour",		Anything,	[ AW, ER2, ] 	],
	[ Anything,		"how",		Anything,	[ HH1, AW, ] 	],
	[ Anything,		"h",		"onor",		Silent	],
	[ Anything,		"h",		"onest",	Silent	],
	[ Anything,		"h",		"#",		[ HH1, ] 	],
	[ Anything,		"h",		Anything,	Silent	],
]

#i 28
r_i = [
	[ Nothing,		"i",		Nothing,	[ AO, AY, ] 	],
	[ Nothing,		"ii",		Nothing,	[ TT2, UW2, ] 	],
	[ Nothing,		"iii",		Nothing,	[ TH, ER1, IY, ] 	],

	[ Nothing,		"intrigu",	"#",		[ IH, NN1, TT2, ER1, IY, GG1, ] 	],
	[ Nothing,		"iso",		Anything,	[ AY, SS, OW, ] 	],
	[ Anything,		"ity",		Nothing,	[ PA1, IH, TT2, IY, ] 	],
	[ Nothing,		"in",		Anything,	[ IH, IH, NN1, ] 	],
	[ Nothing,		"i",		"o",		[ AY, ] 	],
	[ Anything,		"ify",		Anything,	[ PA1, IH, FF, AY, ] 	],
	[ Anything,		"igh",		Anything,	[ AY, ] 	],
	[ Anything,		"ild",		Anything,	[ AY, LL, DD1, ] 	],
	[ Anything,		"ign",		Nothing,	[ AY, NN1, ] 	],
	[ Anything,		"in",		"d",		[ AY, NN1, ] 	],
	[ Anything,		"ier",		Anything,	[ IY, ER2, ] 	],
	[ Anything,		"idea",		Anything,	[ AY, DD2, IY, AX, ] 	],
	[ Nothing,		"idl",		Anything,	[ AY, DD2, AX, LL, ] 	],	#there was previously a 'YYY' at the end
	[ Anything,		"iron",		Anything,	[ AA, AY, ER2, NN1, ] 	],
	[ Anything,		"ible",		Anything,	[ IH, BB1, LL, ] 	],
	[ "r",			"iend",		Anything,	[ AE, NN1, DD1, ] 	],
	[ Anything,		"iend",		Anything,	[ IY, NN1, DD1, ] 	],
	[ "#:r",		"ied",		Anything,	[ IY, DD1, ] 	],
	[ Anything,		"ied",		Nothing,	[ AY, DD1, ] 	],
	[ Anything,		"ien",		Anything,	[ IY, EH, NN1, ] 	],
	[ Anything,		"ion",		Anything,	[ YY2, AX, NN1, ] 	],
	[ "ch",			"ine",		Anything,	[ IY, NN1, ] 	],
	[ "ent",		"ice",		Anything,	[ AY, SS, ] 	],
	[ Anything,		"ice",		Anything,	[ IH, SS, ] 	],
	[ Anything,		"iec",		"%",		[ IY, SS, SS, ] 	],
	[ "#.",			"ies",		Nothing,	[ IY, ZZ, ] 	],
	[ Anything,		"ies",		Nothing,	[ AY, ZZ, ] 	],
	[ Anything,		"ie",		"t",		[ AY, EH, ] 	],
	[ Anything,		"ie",		"^",		[ IY, ] 	],
	[ Anything,		"i",		"cation",	[ IH, ] 	],

	[ Anything,		"ing",		Anything,	[ IH, NG, ] 	],
	[ Anything,		"ign",		"^",		[ AA, AY, NN1, ] 	],
	[ Anything,		"ign",		"%",		[ AA, AY, NN1, ] 	],
	[ Anything,		"ique",		Anything,	[ IY, KK1, ] 	],
	[ Anything,		"ish",		Anything,	[ IH, SH, ] 	],


	[ Nothing,		"ir",		Anything,	[ YR, ] 	],
	[ Anything,		"ir",		"#",		[ AA, AY, ER1, ] 	],
	[ Anything,		"ir",		Anything,	[ ER2, ] 	],
	[ Anything,		"iz",		"%",		[ AA, AY, ZZ, ] 	],
	[ Anything,		"is",		"%",		[ AA, AY, ZZ, ] 	],

	[ "^ch",		"i",		".",		[ AA, AY, ] 	],
	[ "^ch",		"i",		"^",		[ IH, ] 	],
	[ " #^",		"i",		"^",		[ IH, ] 	],
	[ "^#^",		"i",		"^",		[ IH, ] 	],
	[ "^#^",		"i",		"#",		[ IY, ] 	],
	[ ".",			"i",		Nothing,	[ AO, AY, ] 	],
	[ "#^",			"i",		"^#",		[ AY, ] 	],
	[ Anything,		"i",		"gue",		[ IY, ] 	],
	[ ".",			"i",		"ve",		[ AA, AY, ] 	],
	[ Anything,		"i",		"ve",		[ IH, ] 	],
	[ Anything,		"i",		"^+:#",		[ IH, ] 	],
	[ ".",			"i",		"o",		[ AO, AY, ] 	],
	[ "#^",			"i",		"^ ",		[ IH, ] 	],
	[ "#^",			"i",		"^#^",		[ IH, ] 	],
	[ "#^",			"i",		"^",		[ IY, ] 	],
	[ "^",			"i",		"^#",		[ AY, ] 	],
	[ "^",			"i",		"o",		[ IY, ] 	],
	[ ".",			"i",		"a",		[ AY, ] 	],
	[ Anything,		"i",		"a",		[ IY, ] 	],
	[ " :",			"i",		"%",		[ AY, ] 	],
	[ Anything,		"i",		"%",		[ IY, ] 	],
	[ ".",			"i",		".#",		[ AA, AY, ] 	],	# there was previously an 'XX' at the end
	[ Anything,		"i",		"d%",		[ AH, AY, ] 	],
	[ "+^",			"i",		"^+",		[ AH, AY, ] 	],
	[ Anything,		"i",		"t%",		[ AH, AY, ] 	],
	[ "#:^",		"i",		"^+",		[ AH, AY, ] 	],
	[ Anything,		"i",		"^+",		[ AH, AY, ] 	],
	[ ".",			"i",		".",		[ IH, IH, ] 	],
	[ Anything,		"i",		"nus",		[ AA, AY, ] 	],
	[ Anything,		"i",		Anything,	[ IH, ] 	],
]

#j 1
r_j = [
	[ Anything,		"japanese",	Anything,	[ JH, AX, PP, AE, AE, NN1, IY, SS, SS, ] 	],
	[ Anything,		"japan",	Anything,	[ JH, AX, PP, AE, AE, NN1, ] 	],
	[ Anything,		"july",		Anything,	[ JH, UW2, LL, AE, AY, ] 	],
	[ Anything,		"jesus",	Anything,	[ JH, IY, ZZ, AX, SS, ] 	],
	[ Anything,		"j",		Anything,	[ JH, ] 	],
]

#k 2
r_k = [
	[ Nothing,		"k",		"n",		Silent	],

	[ Anything,		"k",		"u",		[ KK3, ] 	],
	[ Anything,		"k",		"o",		[ KK3, ] 	],
	[ Anything,		"k",		"a^^",		[ KK3, ] 	],
	[ Anything,		"k",		"o^^",		[ KK3, ] 	],
	[ Anything,		"k",		"l",		[ KK3, ] 	],
	[ Anything,		"k",		"r",		[ KK3, ] 	],

	[ Anything,		"k",		"a",		[ KK1, ] 	],
	[ Anything,		"k",		"e",		[ KK1, ] 	],
	[ Anything,		"k",		"i",		[ KK1, ] 	],

	[ Anything,		"k",		Nothing,	[ KK2, ] 	],
	[ Anything,		"k",		Anything,	[ KK1, ] 	],
]

#l 5
r_l = [
	[ "l",			"l",		Anything,	Silent	],
	[ Nothing,		"lion",		Anything,	[ LL, AY, AX, NN1, ] 	],
	[ Anything,		"lead",		Anything,	[ LL, IY, DD1, ] 	],
	[ Anything,		"level",	Anything,	[ LL, EH, VV, AX, LL, ] 	],
	[ Anything,		"liber",	Anything,	[ LL, IH, BB2, ER2, ] 	],
	[ Nothing,		"lose",		Anything,	[ LL, UW2, ZZ, ] 	],
	[ Nothing,		"liv",		Anything,	[ LL, IH, VV, ] 	],
	[ "^",			"liv",		Anything,	[ LL, AY, VV, ] 	],
	[ "#",			"liv",		Anything,	[ LL, IH, VV, ] 	],
	[ Anything,		"liv",		Anything,	[ LL, AY, VV, ] 	],
	[ Anything,		"lo",		"c#",		[ LL, OW, ] 	],
	[ "#:^",		"l",		"%",		[ LL, ] 	],

	[ Anything,		"ly",		Nothing,	[ PA1, LL, IY, ] 	],
	[ Anything,		"l",		Anything,	[ LL, ] 	],
]

#m 2
r_m = [
	[ "m",			"m",		Anything,	Silent	],
	[ Nothing,		"my",		Nothing,	[ MM, AY, ] 	],
	[ Nothing,		"mary",		Nothing,	[ MM, EY, XR, IY, ] 	],
	[ "#",			"mary",		Nothing,	[ PA1, MM, EY, XR, IY, ] 	],
	[ Anything,		"micro",	Anything,	[ MM, AY, KK1, ER1, OW, ] 	],
	[ Anything,		"mono",		".",		[ MM, AA, NN1, OW, ] 	],
	[ Anything,		"mono",		"^",		[ MM, AA, NN1, AA, ] 	],
	[ Anything,		"mon",		"#",		[ MM, AA, AA, NN1, ] 	],
	[ Anything,		"mos",		Anything,	[ MM, OW, SS, ] 	],
	[ Anything,		"mov",		Anything,	[ MM, UW2, VV, HH1, ] 	],
	[ "th",			"m",		"#",		[ MM, ] 	],
	[ "th",			"m",		Nothing,	[ IH, MM, ] 	],
	[ Anything,		"m",		Anything,	[ MM, ] 	],
]

#n 8
r_n = [
	[ "n",			"n",		Anything,	Silent	],
	[ Nothing,		"now",		Nothing,	[ NN1, AW, ] 	],
	[ "#",			"ng",		"+",		[ NN1, JH, ] 	],
	[ Anything,		"ng",		"r",		[ NG, GG1, ] 	],
	[ Anything,		"ng",		"#",		[ NG, GG1, ] 	],
	[ Anything,		"ngl",		"%",		[ NG, GG1, AX, LL, ] 	],
	[ Anything,		"ng",		Anything,	[ NG, ] 	],
	[ Anything,		"nk",		Anything,	[ NG, KK1, ] 	],
	[ Nothing,		"none",		Anything,	[ NN2, AH, NN1, ] 	],
	[ Nothing,		"non",		":",		[ NN2, AA, AA, NN1, ] 	],
	[ Anything,		"nuc",		"l",		[ NN2, UW1, KK1, ] 	],

	[ "r",			"n",		Anything,	[ NN1, ] 	],
	[ Anything,		"n",		"#r",		[ NN1, ] 	],
	[ Anything,		"n",		"o",		[ NN2, ] 	],

	[ Anything,		"n",		Anything,	[ NN1, ] 	],
]

#o 48
r_o = [
	[ Nothing,		"only",		Anything,	[ OW, NN1, LL, IY, ] 	],
	[ Nothing,		"once",		Anything,	[ WW, AH, NN1, SS, ] 	],
	[ Nothing,		"oh",		Nothing,	[ OW, ] 	],
	[ Nothing,		"ok",		Nothing,	[ OW, PA3, KK1, EH, EY, ] 	],
	[ Nothing,		"okay",		Nothing,	[ OW, PA3, KK1, EH, EY, ] 	],
	[ Nothing,		"ohio",		Nothing,	[ OW, HH1, AY, OW, ] 	],
	[ Nothing,		"over",		Anything,	[ OW, VV, ER2, ] 	],
	[ Anything,		"other",	Anything,	[ AH, DH2, ER2, ] 	],
	[ Anything,		"ohm",		Nothing,	[ OW, MM, ] 	],

	[ Anything,		"origin",	Anything,	[ OR, IH, DD2, JH, IH, NN1, ] 	],
	[ Anything,		"orough",	Anything,	[ ER2, OW, ] 	],
	[ Anything,		"ought",	Anything,	[ AO, TT2, ] 	],
	[ Anything,		"occu",		"p",		[ AA, KK1, PA1, UW1, ] 	],
	[ Anything,		"ough",		Anything,	[ AH, FF, ] 	],
	[ Anything,		"ore",		Anything,	[ OW, ER1, ] 	],
	[ "#:",			"ors",		Nothing,	[ ER2, ZZ, ] 	],
	[ Anything,		"orr",		Anything,	[ AO, ER1, ] 	],
	[ "d",			"one",		Anything,	[ AH, NN1, ] 	],
	[ "^y",			"one",		Anything,	[ WW, AH, NN1, ] 	],
	[ Nothing,		"one",		Anything,	[ WW, AH, NN1, ] 	],
	[ Anything,		"our",		Nothing,	[ AW, ER1, ] 	],
	[ Anything,		"our",		"^",		[ OR, ] 	],
	[ Anything,		"our",		Anything,	[ AO, AW, ER1, ] 	],
	[ "t",			"own",		Anything,	[ AW, NN1, ] 	],
	[ "br",			"own",		Anything,	[ AW, NN1, ] 	],
	[ "fr",			"own",		Anything,	[ AW, NN1, ] 	],
	[ Anything,		"olo",		Anything,	[ AO, AA, LL, AO, ] 	],
	[ Anything,		"ould",		Anything,	[ UH, DD1, ] 	],
	[ Anything,		"oup",		Anything,	[ UW2, PP, ] 	],
	[ Anything,		"oing",		Anything,	[ OW, IH, NG, ] 	],
	[ Anything,		"omb",		"%",		[ OW, MM, ] 	],
	[ Anything,		"oor",		Anything,	[ AO, ER1, ] 	],
	[ Anything,		"ook",		Anything,	[ UH, KK1, ] 	],
	[ Anything,		"on't",		Anything,	[ OW, NN1, TT2, ] 	],
	[ Anything,		"oss",		Nothing,	[ AO, SS, ] 	],

	[ Anything,		"of",		Nothing,	[ AX, AX, VV, HH1, ] 	],
	[ "^",			"or",		Nothing,	[ AO, AO, ER1, ] 	],
	[ "#:",			"or",		Nothing,	[ ER2, ] 	],
	[ Anything,		"or",		Anything,	[ AO, AO, ER1, ] 	],
	[ Anything,		"ow",		Nothing,	[ OW, ] 	],
	[ Anything,		"ow",		"#",		[ OW, ] 	],
	[ Anything,		"ow",		".",		[ OW, ] 	],
	[ Anything,		"ow",		Anything,	[ AW, ] 	],
	[ " l",			"ov",		Anything,	[ AH, VV, HH1, ] 	],
	[ " d",			"ov",		Anything,	[ AH, VV, HH1, ] 	],
	[ "gl",			"ov",		Anything,	[ AH, VV, HH1, ] 	],
	[ "^",			"ov",		Anything,	[ OW, VV, HH1, ] 	],
	[ Anything,		"ov",		Anything,	[ AH, VV, HH1, ] 	],
	[ Anything,		"ol",		"d",		[ OW, LL, ] 	],
	[ Nothing,		"ou",		Anything,	[ AW, ] 	],
	[ "h",			"ou",		"s#",		[ AW, ] 	],
	[ "ac",			"ou",		"s",		[ UW2, ] 	],
	[ "^",			"ou",		"^l",		[ AH, ] 	],
	[ Anything,		"ou",		Anything,	[ AW, ] 	],
	[ Anything,		"oa",		Anything,	[ OW, ] 	],
	[ Anything,		"oy",		Anything,	[ OY, ] 	],
	[ Anything,		"oi",		Anything,	[ OY, ] 	],
	[ "i",			"on",		Anything,	[ AX, AX, NN1, ] 	],
	[ "#:",			"on",		Nothing,	[ AX, AX, NN1, ] 	],
	[ "#^",			"on",		Anything,	[ AX, AX, NN1, ] 	],
	[ Anything,		"of",		"^",		[ AO, FF, ] 	],
	[ "#:^",		"om",		Anything,	[ AH, MM, ] 	],
	[ Anything,		"oo",		Anything,	[ UW2, ] 	],

	[ Anything,		"ous",		Anything,	[ AX, SS, ] 	],

	[ "^#^",		"o",		"^",		[ AX, ] 	],
	[ "^#^",		"o",		"#",		[ OW, ] 	],
	[ "#",			"o",		".",		[ OW, ] 	],
	[ "^",			"o",		"^#^",		[ AX, AX, ] 	],
	[ "^",			"o",		"^#",		[ OW, ] 	],
	[ Anything,		"o",		"^%",		[ OW, ] 	],
	[ Anything,		"o",		"^en",		[ OW, ] 	],
	[ Anything,		"o",		"^i#",		[ OW, ] 	],
	[ Anything,		"o",		"e",		[ OW, ] 	],
	[ Anything,		"o",		Nothing,	[ OW, ] 	],
	[ "c",			"o",		"n",		[ AA, ] 	],
	[ Anything,		"o",		"ng",		[ AO, ] 	],
	[ " :^",		"o",		"n",		[ AX, ] 	],
	[ Anything,		"o",		"st ",		[ OW, ] 	],
	[ Anything,		"o",		Anything,	[ AO, ] 	],
]

#p 5
r_p = [
	[ Nothing,		"pi",		Nothing,	[ PP, AY, ] 	],
	[ Anything,		"put",		Nothing,	[ PP, UH, TT2, ] 	],
	[ Anything,		"prove",	Anything,	[ PP, ER1, UW2, VV, ] 	],
	[ Anything,		"ply",		Anything,	[ PP, LL, AY, ] 	],
	[ "p",			"p",		Anything,	Silent	],
	[ Anything,		"phe",		Nothing,	[ FF, IY, ] 	],
	[ Anything,		"phe",		"s ",		[ FF, IY, ] 	],
	[ Anything,		"peop",		Anything,	[ PP, IY, PP, ] 	],
	[ Anything,		"pow",		Anything,	[ PP, AW, ] 	],
	[ Anything,		"ph",		Anything,	[ FF, ] 	],
	[ Anything,		"p",		Anything,	[ PP, ] 	],
]

#q 3
r_q = [
	[ Anything,		"quar",		Anything,	[ KK3, WW, AO, ER1, ] 	],
	[ Anything,		"que",		Nothing,	[ KK2, ] 	],
	[ Anything,		"que",		"s",		[ KK2, ] 	],
	[ Anything,		"qu",		Anything,	[ KK3, WW, ] 	],
	[ Anything,		"q",		Anything,	[ KK1, ] 	],
]

#r 2
r_r = [
	[ Nothing,		"rugged",	Anything,	[ ER1, AX, GG1, PA1, EH, DD1, ] 	],
	[ Nothing,		"russia",	Anything,	[ ER1, AX, SH, PA1, AX, ] 	],
	[ Nothing,		"reality",	Anything,	[ ER1, IY, AE, LL, IH, TT2, IY, ] 	],
	[ Anything,		"radio",	Anything,	[ ER1, EY, DD2, IY, OW, ] 	],
	[ Anything,		"radic",	Anything,	[ ER1, AE, DD2, IH, KK1, ] 	],
	[ Nothing,		"re",		"^#",		[ ER1, IY, ] 	],
	[ Nothing,		"re",		"^^#",		[ ER1, IY, ] 	],
	[ Nothing,		"re",		"^^+",		[ ER1, IY, ] 	],

	[ "^",			"r",		Anything,	[ RR2, ] 	],
	[ Anything,		"r",		Anything,	[ ER1, ] 	],
]

#s 23
r_s = [
	[ Anything,		"said",		Anything,	[ SS, EH, DD1, ] 	],
	[ Anything,		"secret",	Anything,	[ SS, IY, KK1, ER1, EH, TT2, ] 	],
	[ Nothing,		"sly",		Anything,	[ SS, LL, AY, ] 	],
	[ Nothing,		"satur",	Anything,	[ SS, AE, AE, TT2, ER2, ] 	],
	[ Anything,		"some",		Anything,	[ SS, AH, MM, ] 	],
	[ Anything,		"s",		"hon#^",	[ SS, ] 	],
	[ Anything,		"sh",		Anything,	[ SH, ] 	],
	[ "#",			"sur",		"#",		[ ZH, ER2, ] 	],
	[ Anything,		"sur",		"#",		[ SH, ER2, ] 	],
	[ "#",			"su",		"#",		[ ZH, UW2, ] 	],
	[ "#",			"ssu",		"#",		[ SH, UW2, ] 	],
	[ "#",			"sed",		Nothing,	[ ZZ, DD1, ] 	],
	[ "#",			"sion",		Anything,	[ PA1, ZH, AX, NN1, ] 	],
	[ "^",			"sion",		Anything,	[ PA1, SH, AX, NN1, ] 	],
	[ "s",			"sian",		Anything,	[ SS, SS, IY, AX, NN1, ] 	],
	[ "#",			"sian",		Anything,	[ PA1, ZH, IY, AX, NN1, ] 	],
	[ Anything,		"sian",		Anything,	[ PA1, ZH, AX, NN1, ] 	],
	[ Nothing,		"sch",		Anything,	[ SS, KK1, ] 	],
	[ "#",			"sm",		Anything,	[ ZZ, MM, ] 	],
	[ "#",			"sn",		"'",		[ ZZ, AX, NN1, ] 	],
	[ Nothing,		"sky",		Anything,	[ SS, KK1, AY ] 	],
	[ "#",			"s",		"#",		[ ZZ, ] 	],
	[ ".",			"s",		Nothing,	[ ZZ, ] 	],
	[ "#:.e",		"s",		Nothing,	[ ZZ, ] 	],
	[ "#:^##",		"s",		Nothing,	[ ZZ, ] 	],
	[ "#:^#",		"s",		Nothing,	[ SS, ] 	],
	[ "u",			"s",		Nothing,	[ SS, ] 	],
	[ " :#",		"s",		Nothing,	[ ZZ, ] 	],
	[ Anything,		"s",		"s",		Silent	],
	[ Anything,		"s",		"c+",		Silent	],
	[ Anything,		"s",		Anything,	[ SS, ] 	],
]

#t 26
r_t = [
	[ Nothing,		"the",		Nothing,	[ DH1, IY, ] 	],
	[ Nothing,		"this",		Nothing,	[ DH2, IH, IH, SS, SS, ] 	],
	[ Nothing,		"than",		Nothing,	[ DH2, AE, AE, NN1, ] 	],
	[ Nothing,		"them",		Nothing,	[ DH2, EH, EH, MM, ] 	],
	[ Nothing,		"tilde",	Nothing,	[ TT2, IH, LL, DD2, AX, ] 	],
	[ Nothing,		"tuesday",	Nothing,	[ TT2, UW2, ZZ, PA2, DD2, EY, ] 	],

	[ Nothing,		"try",		Anything,	[ TT2, ER1, AY, ] 	],
	[ Nothing,		"thy",		Anything,	[ DH2, AY, ] 	],
	[ Nothing,		"they",		Anything,	[ DH2, EH, EY, ] 	],
	[ Nothing,		"there",	Anything,	[ DH2, EH, XR, ] 	],
	[ Nothing,		"then",		Anything,	[ DH2, EH, EH, NN1, ] 	],
	[ Nothing,		"thus",		Anything,	[ DH2, AH, AH, SS, ] 	],
	[ Anything,		"that",		Nothing,	[ DH2, AE, TT2, ] 	],

	[ Anything,		"truly",	Anything,	[ TT2, ER1, UW2, LL, IY, ] 	],
	[ Anything,		"truth",	Anything,	[ TT2, ER1, UW2, TH, ] 	],
	[ Anything,		"their",	Anything,	[ DH2, EH, IY, XR, ] 	],
	[ Anything,		"these",	Nothing,	[ DH2, IY, ZZ, ] 	],
	[ Anything,		"through",	Anything,	[ TH, ER1, UW2, ] 	],
	[ Anything,		"those",	Anything,	[ DH2, OW, ZZ, ] 	],
	[ Anything,		"though",	Nothing,	[ DH2, OW, ] 	],

	[ Anything,		"tion",		Anything,	[ PA1, SH, AX, NN1, ] 	],
	[ Anything,		"tian",		Anything,	[ PA1, SH, AX, NN1, ] 	],
	[ Anything,		"tien",		Anything,	[ SH, AX, NN1, ] 	],

	[ Anything,		"tear",		Nothing,	[ TT2, EY, ER2, ] 	],
	[ Anything,		"tear",		"%",		[ TT2, EY, ER2, ] 	],
	[ Anything,		"tear",		"#",		[ TT2, EY, ER2, ] 	],

	[ "#",			"t",		"ia",		[ SH, ] 	],
	[ ".",			"t",		"ia",		[ SH, ] 	],

	[ Anything,		"ther",		Anything,	[ DH2, PA2, ER2, ] 	],
	[ Anything,		"to",		Nothing,	[ TT2, UW2, ] 	],
	[ "#",			"th",		Anything,	[ TH, ] 	],
	[ Anything,		"th",		Anything,	[ TH, ] 	],
	[ "#:",			"ted",		Nothing,	[ PA1, TT2, IH, DD1, ] 	],
	[ Anything,		"tur",		"#",		[ PA1, CH, ER2, ] 	],
	[ Anything,		"tur",		"^",		[ TT2, ER2, ] 	],
	[ Anything,		"tu",		"a",		[ CH, UW2, ] 	],
	[ Nothing,		"two",		Anything,	[ TT2, UW2, ] 	],

	[ "t",			"t",		Anything,	Silent	],

	[ Anything,		"t",		"s",		[ TT1, ] 	],
	[ Anything,		"t",		Anything,	[ TT2, ] 	],
]

#u 35
r_u = [
	[ Nothing,		"un",		Nothing,	[ YY2, UW2, PA3, AE, NN1, ] 	],
	[ Nothing,		"usa",		Nothing,	[ YY2, UW2, PA3, AE, SS, SS, PA3, EH, EY, ] 	],
	[ Nothing,		"ussr",		Nothing,	[ YY2, UW2, PA3, AE, SS, SS, PA3, AE, SS, SS, PA3, AA, AR, ] 	],

	[ Nothing,		"u",		Nothing,	[ YY2, UW1, ] 	],
	[ Nothing,		"un",		"i",		[ YY2, UW2, NN1, ] 	],
	[ Nothing,		"un",		":",		[ AH, NN1, PA1, ] 	],
	[ Nothing,		"un",		Anything,	[ AH, NN1, ] 	],
	[ Nothing,		"upon",		Anything,	[ AX, PP, AO, NN1, ] 	],
	[ "d",			"up",		Anything,	[ UW2, PP, ] 	],
	#[ Anything,		"use",		".",		[ UW1, ZZ, ] 	],
	[ "t",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "s",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "r",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "d",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "l",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "z",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "n",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "j",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "th",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "ch",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "sh",			"ur",		"#",		[ UH, ER1, ] 	],
	[ "arg",		"u",		"#",		[ YY2, UW2, ] 	],
	[ Anything,		"ur",		"#",		[ YY2, UH, ER1, ] 	],
	[ Anything,		"ur",		Anything,	[ ER2, ] 	],
	[ Anything,		"uy",		Anything,	[ AA, AY, ] 	],

	[ Anything,		"u",		"^#^",		[ YY2, UW2, ] 	],
	[ Anything,		"u",		"^ ",		[ AH, ] 	],
	[ Anything,		"u",		"%",		[ UW2, ] 	],
	[ " g",			"u",		"#",		Silent	],
	[ "g",			"u",		"%",		Silent	],
	[ "g",			"u",		"#",		[ WW, ] 	],
	[ "#n",			"u",		Anything,	[ YY2, UW2, ] 	],
	[ "#m",			"u",		Anything,	[ YY2, UW2, ] 	],
	[ "f",			"u",		"^^",		[ UH, ] 	],
	[ "b",			"u",		"^^",		[ UH, ] 	],
	[ "^",			"u",		"^e",		[ YY2, UW2, ] 	],
	[ "^",			"u",		"^",		[ AX, ] 	],
	[ Anything,		"u",		"^^",		[ AH, ] 	],
	[ "t",			"u",		Anything,	[ UW2, ] 	],
	[ "s",			"u",		Anything,	[ UW2, ] 	],
	[ "r",			"u",		Anything,	[ UW2, ] 	],
	[ "d",			"u",		Anything,	[ UW2, ] 	],
	[ "l",			"u",		Anything,	[ UW2, ] 	],
	[ "z",			"u",		Anything,	[ UW2, ] 	],
	[ "n",			"u",		Anything,	[ UW2, ] 	],
	[ "j",			"u",		Anything,	[ UW2, ] 	],
	[ "th",			"u",		Anything,	[ UW2, ] 	],
	[ "ch",			"u",		Anything,	[ UW2, ] 	],
	[ "sh",			"u",		Anything,	[ UW2, ] 	],
	[ Anything,		"u",		Anything,	[ YY2, UW2, ] 	],
]

#v 2
r_v = [
	[ Anything,		"view",		Anything,	[ VV, YY2, UW2, ] 	],
	[ Nothing,		"very",		Nothing,	[ VV, EH, ER2, PA1, IY, ] 	],
	[ Anything,		"vary",		Anything,	[ VV, EY, PA1, ER1, IY, ] 	],
	[ Anything,		"v",		Anything,	[ VV, ] 	],
]

#w 12
r_w = [
	[ Nothing,		"were",		Anything,	[ WW, ER2, ] 	],
	[ Anything,		"weigh",	Anything,	[ WW, EH, EY, ] 	],
	[ Anything,		"wood",		Anything,	[ WW, UH, UH, DD1, ] 	],
	[ Anything,		"wary",		Anything,	[ WW, EH, ER2, PA1, IY, ] 	],
	[ Anything,		"where",	Anything,	[ WW, EH, ER1, ] 	],
	[ Anything,		"what",		Anything,	[ WW, AA, AA, TT2, ] 	],
	[ Anything,		"want",		Anything,	[ WW, AA, AA, NN1, TT2, ] 	],
	[ Anything,		"whol",		Anything,	[ HH1, OW, LL, ] 	],
	[ Anything,		"who",		Anything,	[ HH1, UW2, ] 	],
	[ Anything,		"why",		Anything,	[ WW, AO, AY, ] 	],
	[ Anything,		"wear",		Anything,	[ WW, EY, ER2, ] 	],
	[ Anything,		"wea",		"th",		[ WW, EH, ] 	],
	[ Anything,		"wea",		"l",		[ WW, EH, ] 	],
	[ Anything,		"wea",		"p",		[ WW, EH, ] 	],
	[ Anything,		"wa",		"s",		[ WW, AA, ] 	],
	[ Anything,		"wa",		"t",		[ WW, AA, ] 	],
	[ Anything,		"wh",		Anything,	[ WH, ] 	],
	[ Anything,		"war",		Nothing,	[ WW, AO, ER1, ] 	],
	[ Nothing,		"wicked",	Anything,	[ WW, IH, KK2, PA1, EH, DD1, ] 	],
	[ "be",			"wilder",	Anything,	[ WW, IH, LL, DD2, ER2, ] 	],
	[ Nothing,		"wilder",	"ness",		[ WW, IH, LL, DD2, ER2, ] 	],
	[ Nothing,		"wild",		"erness",	[ WW, IH, LL, DD2, ] 	],
	[ Nothing,		"wily",		Nothing,	[ WW, AY, LL, IY, ] 	],
	[ Anything,		"wor",		"^",		[ WW, ER2, ] 	],
	[ Anything,		"wr",		Anything,	[ ER1, ] 	],

	[ Anything,		"w",		Anything,	[ WW, ] 	],
]

#x 1
r_x = [
	[ Anything,		"x",		Anything,	[ KK1, SS, ] 	],
]

#y 11
r_y = [
	[ Anything,		"young",	Anything,	[ YY2, AH, NG, ] 	],
	[ Nothing,		"your",		Anything,	[ YY2, UH, ER2, ] 	],
	[ Nothing,		"you",		Anything,	[ YY2, UW2, ] 	],
	[ Nothing,		"yes",		Anything,	[ YY2, EH, SS, ] 	],
	[ Anything,		"yte",		Anything,	[ AY, TT2, PA1, ] 	],

	[ Anything,		"y",		Nothing,	[ IY, ] 	],
	[ Anything,		"y",		Anything,	[ IH, ] 	],
	#[ Nothing,		"y",		Anything,	[ YY2, ] 	],
	#[ "ph",			"y",		Anything,	[ IH, ] 	] 
	#[ ":s",			"y",		".",		[ IH, ] 	] 
	#[ "#^",			"y",		".",		[ AY, ] 	] 
	#[ "h",			"y",		"^",		[ AY, ] 	] 
	#[ "#",			"y",		"#",		[ OY, ] 	] 
	#[ "^",			"y",		"z",		[ AY, ] 	],
	#[ "#:^",		"y",		Nothing,	[ IY, ] 	],
	#[ "#:^",		"y",		"i",		[ IY, ] 	],
	#[ " :",			"y",		Nothing,	[ AY, ] 	],
	#[ " :",			"y",		"#",		[ AY, ] 	],
	#[ " :",			"y",		".",		[ AY, ] 	],
	#[ " :",			"y",		"^+:#",		[ IH, ] 	],
	#[ " :",			"y",		"^#",		[ AY, ] 	],
	[ Anything,		"y",		Anything,	[ IH, ] 	],
]

#z 1
r_z = [
	[ "z",			"z",		Anything,	Silent	],
	[ Anything,		"z",		Anything,	[ ZZ, ] 	],
]



_rules = [
	r_punc,
	r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h,
	r_i, r_j, r_k, r_l, r_m, r_n, r_o, r_p,
	r_q, r_r, r_s, r_t, r_u, r_v, r_w, r_x,
	r_y, r_z,
]




#i.e., not is punctuation
def _isAlpha ( ch ):
	return ( ch >= 'a' and ch <= 'z' )


#'#'
def _isVowel ( ch ):
	return ( 'a' == ch or 'e' == ch or 'i' == ch or 'o' == ch or 'u' == ch or 'y' == ch )


#'*' one or more consonants; also used for '^' (one consonant), and ':' (zero or more)
def _isConsonant ( ch ):
	return ( 'b' == ch or 'c' == ch or 'd' == ch or 'f' == ch or 'g' == ch or 
		'h' == ch or 'j' == ch or 'k' == ch or 'l' == ch or 'm' == ch or 
		'n' == ch or 'p' == ch or 'q' == ch or 'r' == ch or 's' == ch or 
		't' == ch or 'v' == ch or 'w' == ch or 'x' == ch or #'y' == ch or 
		'z' == ch )


#'.'
def _isVoicedConsonant ( ch ):
	return ( 'b' == ch or 'd' == ch or 'g' == ch or 'j' == ch or 'l' == ch or 
		'm' == ch or 'n' == ch or 'r' == ch or 'v' == ch or 'w' == ch or 
		'z' == ch )


#'+'
def _isFrontVowel ( ch ) :
	return ( 'e' == ch or 'i' == ch or 'y' == ch )




#I am writing this without using regexes simply because there will not be that
#capability on the final target, and I'm hoping to make this code more-or-less
#directly translatable to such.

#word := 
#  one or more letter sequences with apostrophe:  [a-zA-Z']+
#  (this is a basic translatable word)
# or
#  one or more punctuation sequence: [/,:;!-\.\?]+
#  (this is an internal word separator that may be adjacent-to/inside a word)
# else skip this character

#so, really there are three character classes to consider

def _classifyChar ( ch ):
	if ( ( ch >= 'A' and ch <= 'Z' ) or ( ch >= 'a' and ch <= 'z' ) or ( ch == "'" ) ):
		return 1
	elif ( ch == '/' or ch == ',' or ch == ':' or ch == ';' or ch == '!' or ch == '-' or ch == '.' or ch == '?' ):
		return 2
	else:
		return 0



def _matchLeft ( strNormWord, nIdxEnd, strCtx ):
	if (None == strCtx  or 0 == len(strCtx)):	#'anything'? (empty string)
		return True
	
	#OK we match this backwards from the end
	nIdxText = nIdxEnd - 1;			#last char in text
	nIdxMatch = len(strCtx) - 1;	#last char in pattern
	#whiz over the context characters, consuming input from the end
	while ( nIdxMatch >= 0 ):
		#try literals
		chThisCtx = strCtx[nIdxMatch];
		if ( _isAlpha ( chThisCtx ) or '\'' == chThisCtx or ' ' == chThisCtx ):
			if ( chThisCtx != strNormWord[nIdxText] ):
				return False	#fail; done
			#consume input and carry on
			nIdxText -= 1
		else:
			#must be a pattern metachar
			if ( chThisCtx == '#' ):	#one or more vowels
				if ( not _isVowel ( strNormWord[nIdxText] ) ):
					return False;
				nIdxText -= 1
				while ( _isVowel ( strNormWord[nIdxText] ) ):
					nIdxText -= 1
			elif ( chThisCtx == ':' ):	#zero or more consonants
				while ( _isConsonant ( strNormWord[nIdxText] ) ):
					nIdxText -= 1
			elif ( chThisCtx == '^' ):	#one consonant
				if ( not _isConsonant ( strNormWord[nIdxText] ) ):
					return False
				nIdxText -= 1
			elif ( chThisCtx == '.' ):	#one voiced consonant
				if ( not _isVoicedConsonant ( strNormWord[nIdxText] ) ):
					return False;
				nIdxText -= 1
			elif ( chThisCtx == '+' ):	#one front vowel
				if ( not _isFrontVowel ( strNormWord[nIdxText] ) ):
					return False
				nIdxText -= 1
			else:	#'%' can't be in left context
				return False;

		nIdxMatch -= 1

	return True



def _matchRight ( strNormWord, nIdxStart, strCtx ):
	if ( None == strCtx or 0 == len(strCtx) ):	#'anything'? (empty string)
		return True
	if ( nIdxStart >= len(strNormWord) ):	#empty text can match nothing
		return False

	#OK we match this forwards from the beginning
	nIdxText = nIdxStart	#first char in text
	nIdxMatch = 0			#first char in pattern
	#whiz over the context characters, consuming input from the beginning
	while ( nIdxMatch < len(strCtx) ):
		#try literals
		chThisCtx = strCtx[nIdxMatch]
		if ( _isAlpha ( chThisCtx ) or '\'' == chThisCtx or ' ' == chThisCtx ):
			if ( chThisCtx != strNormWord[nIdxText] ):
				return False	#fail; done
			#consume input and carry on
			nIdxText += 1
		else:
			#must be a pattern metachar
			if ( chThisCtx == '#' ):	#one or more vowels
				if ( not _isVowel(strNormWord[nIdxText]) ):
					return False
				nIdxText += 1
				while ( _isVowel(strNormWord[nIdxText]) ):
					nIdxText += 1
			elif ( chThisCtx == ':' ):	#zero or more consonants
				while ( _isConsonant(strNormWord[nIdxText]) ):
					nIdxText += 1
			elif ( chThisCtx == '^' ):	#one consonant
				if ( not _isConsonant(strNormWord[nIdxText]) ):
					return False
				nIdxText += 1
			elif ( chThisCtx == '.' ):	#one voiced consonant
				if ( not _isVoicedConsonant(strNormWord[nIdxText]) ):
					return False
				nIdxText += 1
			elif ( chThisCtx == '+' ):	#once front vowel
				if ( not _isFrontVowel(strNormWord[nIdxText]) ):
					return False
				nIdxText += 1
			elif ( chThisCtx == '%' ):	#'e'-related things at the end of the word '-e', '-ed', '-er', '-es', '-ely', '-ing'
				if ( 'e' == strNormWord[nIdxText] ):
					nIdxText += 1	#we will definitely take the e; now see if we can also consume an ly, r, s, or d
					if ( 'l' == strNormWord[nIdxText] ):
						nIdxText += 1
						if ( 'y' == strNormWord[nIdxText] ):
							nIdxText += 1
						else:
							nIdxText -= 1;	#don't consume the 'l'
					elif ( 'r' == strNormWord[nIdxText] or 's' == strNormWord[nIdxText] or 'd' == strNormWord[nIdxText] ):
						nIdxText += 1
				elif ( 'i' == strNormWord[nIdxText]) :
					nIdxText += 1
					if ( 'n' == strNormWord[nIdxText] ):
						nIdxText += 1
						if ( 'g' == strNormWord[nIdxText] ):
							nIdxText += 1
						else:
							return False
				else:
					return False
				
			else:	#horror unknown
				return False;

		nIdxMatch += 1

	return True;



def _transforminput ( strNormWord, nIdx, nIdxRuleSect, phonemes ):
	nConsumed = 1;	#we'll figure it out, but must always consume something

	for rule in _rules[nIdxRuleSect]:
		#first, see if the 'bracket' context matches, by scanning forward
		nIdxText = nIdx;
		nIdxMatch = 0;
		while ( nIdxText < len(strNormWord) and nIdxMatch < len(rule[1]) ):
			#XXX must we consider metachars in match rule?
			if (strNormWord[nIdxText] != rule[1][nIdxMatch]):
				break
			nIdxText += 1
			nIdxMatch += 1
		#if we didn't match all of the pattern, then it is not a match
		if ( nIdxMatch != len(rule[1]) ):
			continue
		#see if the left context matches
		if ( not _matchLeft ( strNormWord, nIdx, rule[0] ) ):
			continue;
		#see if the right context matches
		if ( not _matchRight ( strNormWord, nIdxText, rule[2] ) ):
			continue;
		#match! push the associated phoneme sequence, and update what we have consumed
		phonemes.extend ( rule[3] )
		nConsumed = nIdxText - nIdx
		break

	return nConsumed	#MUST consume some



def _texttospeechword ( strWord, phonemes ):
	#print ( strWord )
	#fix the word up for searching by padding and transforming to consistent (lower) case
	strNormWord = " " + strWord.lower() + " ";

	#scan the juicy bits
	nIdx = 1;	#(skipping past the lead padding)
	while ( nIdx < len(strNormWord) ):
		#use the first character to skip to a section of rules
		chNow = strNormWord[nIdx];
		nIdxRuleSect = 0
		if ( _isAlpha(chNow) ):
			nIdxRuleSect = ord(chNow) - ord('a') + 1
		else:
			nIdxRuleSect = 0

		#whiz through rules to find a match, consume input.  must consume some!
		nIdx += _transforminput ( strNormWord, nIdx, nIdxRuleSect, phonemes );



def _texttospeech ( strText, phonemes ):
	#crack str into words and text-to-speech them
	nIdxStart = 0
	nIdxEnd = 0

	while ( nIdxStart < len(strText) ) :

		#print ( "skipping..." )

		#skip leading non-chars
		while ( nIdxStart < len(strText) and 
				0 == _classifyChar ( strText[nIdxStart] ) ):
			nIdxStart += 1

		#print ( "gathering..." )
		#gather until word break (or end)
		nIdxEnd = nIdxStart
		while ( nIdxEnd < len(strText) and 
				0 != _classifyChar ( strText[nIdxEnd] ) ):
			nIdxEnd += 1

		#print ( "nIdxStart: " + str(nIdxStart) + ", nIdxEnd: " + str(nIdxEnd) + ", ", end = '' )
		#translate word
		_texttospeechword ( strText[nIdxStart:nIdxEnd], phonemes )

		#next word
		nIdxStart = nIdxEnd



strText = R"""Hello, I love you
Won't you tell me your name?"""

phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'hello.wav', phonemes )




strText = R"""Hello, I love you
Won't you tell me your name?
Hello, I love you
Let me jump in your game
Hello, I love you
Won't you tell me your name?
Hello, I love you
Let me jump in your game

She's walking down the street
Blind to every eye she meets
Do you think you'll be the guy
To make the queen of the angels sigh?

Hello, I love you
Won't you tell me your name?
Hello, I love you
Let me jump in your game
Hello, I love you
Won't you tell me your name?
Hello, I love you
Let me jump in your game

She holds her head so high
Like a statue in the sky
Her arms are wicked, and her legs are long
When she moves my brain screams out this song

Sidewalk crouches at her feet
Like a dog that begs for something sweet
Do you hope to make her see, you fool?
Do you hope to pluck this dusky jewel?

Hello, Hello, Hello, Hello, Hello, Hello, Hello
I want you
Hello
I need my baby
Hello, Hello, Hello, Hello"""


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'hello.wav', phonemes )





strText = R"""You need cooling, baby, I'm not fooling
I'm gonna send you back to schooling
Way down inside, honey, you need it
I'm gonna give you my love
I'm gonna give you my love, oh

Wanna whole lotta love
Wanna whole lotta love
Wanna whole lotta love
Wanna whole lotta love

You've been yearning
And baby, I been burning
All them good times
Baby, baby, I've been discerning
way, way down inside
honey, you need
I'm gonna give you my love, ah
I'm gonna give you my love, ah

Oh, whole lotta love
Wanna whole lotta love
Wanna whole lotta love
Wanna whole lotta love
I don't want more

You been cooling
And baby, I've been drooling
All the good times, baby, I've been misusing
way, way down inside
I'm gonna give you my love
I'm gonna give you every inch of my love
I'm gonna give you my love
Yes, alright, let's go

Wanna whole lotta love
Wanna whole lotta love
Wanna whole lotta love
Wanna whole lotta love

Way down inside, woman, you need it
Love

My, my, my, my
My, my, my, my
Oh, shake for me, girl
I wanna be your backdoor man
Hey, oh, hey, oh
Hey, oh, oooh
Oh, oh, oh, oh
Keep a cooling, baby
keep a cooling, baby
keep a cooling, baby
Uh, keep a cooling, baby"""


phonemes = []
_texttospeech ( strText, phonemes )
whizanddump ( 'wholelottalove.wav', phonemes )




strText = R"""When the truth is found to be lies
And all the joy within you dies

Don't you want somebody to love
Don't you need somebody to love
Wouldn't you love somebody to love
You better find somebody to love

When the garden's flowers, baby, are dead
Yes, and your mind, your mind is so full of red

Don't you want somebody to love
Don't you need somebody to love
Wouldn't you love somebody to love
You better find somebody to love

Your eyes, I say your eyes may look like his
Yeah but in your head, baby
I'm afraid you don't know where it is

Don't you want somebody to love
Don't you need somebody to love
Wouldn't you love somebody to love
You better find somebody to love";

Tears are running, they're all running down your breast
And your friends, baby, they treat you like a guest"""


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'somebodytolove.wav', phonemes )




strText = R"""Four score and seven years ago our fathers brought forth on this continent 
a new nation, conceived in liberty, and dedicated to the proposition that 
all men are created equal.
Now we are engaged in a great civil war, testing whether that nation, or any 
nation so conceived and so dedicated, can long endure. We are met on a great 
battlefield of that war. We have come to dedicate a portion of that field, as 
a final resting place for those who here gave their lives that that nation 
might live. It is altogether fitting and proper that we should do this.
But, in a larger sense, we can not dedicate, we can not consecrate, we can not 
hallow this ground. The brave men, living and dead, who struggled here, have 
consecrated it, far above our poor power to add or detract. The world will 
little note, nor long remember what we say here, but it can never forget what 
they did here. It is for us the living, rather, to be dedicated here to the 
unfinished work which they who fought here have thus far so nobly advanced. 
It is rather for us to be here dedicated to the great task remaining before 
us -- that from these honored dead we take increased devotion to that cause 
for which they gave the last full measure of devotion -- that we here highly 
resolve that these dead shall not have died in vain -- that this nation, under 
God, shall have a new birth of freedom -- and that government of the people, by 
the people, for the people, shall not perish from the earth."""


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'gettysburg.wav', phonemes )




strText = R"""It would be a considerable invention indeed, that of
a machine able to mimic our speech, with its sounds
and articulations. I think it is not impossible."""
#- Leonhard Euler, 1761


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'euler.wav', phonemes )




#naked lunch
strText = R"""In the City Market is the Meet Cafe. Followers of obsolete, unthinkable
trades doodling in Estruscan, addicts of drugs not yet synthesized,
pushers of souped up Harmaline, junk reduced to pure habit offering
precarious vegetable serenity, liquids to induce Latah, Tithonian
longevity serums, black marketeers of World War III, excisors of
telepathic sensitivity, osteopaths of the spirit, investigators of
infractions denounced by bland paranoid chess players, servers of
fragmentary warrants taken down in hebephrenic shorthand charging
unspeakable mutilations of the spirit, bureaucrats of spectral
departments, officials of unconstituted police states, a Lesbian dwarf
who has perfected operation Bang-utot, the lung erection that
strangles a sleeping enemy, sellers of orgone tanks and relaxing
machines, brokers of exquisite dreams and memories tested on the
sensitized cells of junk sickness and bartered for raw materials of the
will, doctors skilled in the treatment of diseases dormant in the black
dust of ruined cities, gathering virulence in the white blood of eyeless
worms feeling slowly to the surface and the human host, maladies of the
ocean floor and the stratosphere, maladies of the laboratory and atomic
war... A place where the unknown past and the emergent future meet in a
vibrating soundless hum... Larval entities waiting for a Live One..."""


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'nakedlunch.wav', phonemes )




#jabberwocky
strText = R"""'Twas brillig, and the slithy toves
	Did gyre and gimble in the wabe:
All mimsy were the borogoves,
	And the mome raths outgrabe.

Beware the Jabberwock, my son!
	The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
	The frumious Bandersnatch!

He took his vorpal sword in hand:
	Long time the manxome foe he sought--
So rested he by the Tumtum tree,
	And stood awhile in thought. 

And, as in uffish thought he stood,
	The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
	And burbled as it came! 

One two! One two! And through and through
	The vorpal blade went snicker-snack!
He left it dead, and with its head
	He went galumphing back. 

And hast thou slain the Jabberwock?
	Come to my arms, my beamish boy!
O frabjous day! Callooh! Callay!
	He chortled in his joy. 

'Twas brillig, and the slithy toves
	Did gyre and gimble in the wabe:
All mimsy were the borogoves,
	And the mome raths outgrabe."""


phonemes = []
_texttospeech ( strText, phonemes )
#whizanddump ( 'jabberwocky.wav', phonemes )




print ( 'finishing' )
