#
#
#

from nltk.tokenize import word_tokenize, sent_tokenize
from explore import filter_tokens
import re, string

a = [ word_tokenize("Oh my oh my look at all these tokens!"),
      word_tokenize("Now there's some more punctuation in here!"),
      word_tokenize("Even! More! punctuation! Oh! boy! ! !"),
      word_tokenize("No I've got a lot's of punct'tn, ya' here?!"),]

r = re.compile('[^%s]+' % re.escape(string.punctuation))

for item in a:
    test = filter_tokens(item, r)
    print(test)
print("------------------------")
b = sent_tokenize("We got this GPS for my husband who is an (OTR) over the road trucker.  Very Impressed with the "
                  "shipping time, it arrived a few days earlier than expected...  within a week of use however it "
                  "started freezing up... could of just been a glitch in that unit.  Worked great when it worked!  Will"
                  " work great for the normal person as well but does have the \"trucker\" option. (the big truck "
                  "routes - tells you when a scale is coming up ect...)  Love the bigger screen, \the ease of use, the "
                  "ease of putting addresses into memory.  Nothing really bad to say about the unit with\
                  the exception of it freezing which is probably one in a million and that's just my luck.  I contacted "
                  "the seller and within minutes of my email I received a email back with instructions for an "
                  "exchange! VERY impressed all the way around!")
b = [ word_tokenize(sent) for sent in b ]
print(b)
print("*")
for sent in b:
    test = filter_tokens(sent, r)
    print(test)