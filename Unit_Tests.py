#
#
#

from nltk.tokenize import word_tokenize, sent_tokenize
from helpers import filter_tokens
import re, string
from helpers import *

a = [word_tokenize("Oh my oh my look at all these tokens!"),
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

print("------------------------")
###########################

some_probs = [[  1.12890977e-03,   4.65590794e-04,   2.75452335e-03,   5.74420933e-02,
    9.38208883e-01],
 [  1.33719984e-03,   1.44869620e-03,   6.06645536e-03,   4.97107330e-02,
    9.41436916e-01],
 [  2.41855425e-03,  1.18110676e-03,   3.31689653e-03,   6.93514033e-02,
    9.23732039e-01],
 [  8.70052532e-03 ,  5.35338043e-03 ,  1.16182647e-02,   1.30565332e-01,
    8.43762497e-01],
 [  3.73600704e-03 ,  2.06739681e-03  , 1.11842173e-02,   6.87876680e-02,
    9.14224711e-01],
 [  1.33765350e-02 ,  6.29184959e-03  , 2.18554430e-02,   9.89298860e-02,
    8.59546286e-01],
 [  2.71515947e-04   ,2.52102672e-04 ,  2.19484953e-03  , 5.70043843e-02,
    9.40277148e-01],
 [  2.89452033e-03  , 1.45103177e-03  , 8.50424161e-03  , 1.36842723e-01,
    8.50307483e-01],
 [  5.35347786e-03 ,  3.10967761e-03  , 1.56444355e-02  , 1.29367688e-01,
    8.46524721e-01],
 [  1.33412862e-03   ,6.41653720e-04   ,3.74673259e-03  , 5.70384514e-02,
    9.37239034e-01],
 [  6.04201620e-04   ,7.17682764e-04  , 4.22709134e-03  , 5.94362042e-02,
    9.35014820e-01],
 [  1.88645700e-03  , 1.47892992e-03  , 6.61725991e-03  , 7.87989502e-02,
    9.11218403e-01],
 [  1.12146997e-04 ,  6.78220345e-05 ,  6.14774586e-04  , 2.91485240e-02,
    9.70056732e-01],
 [  4.02404617e-03 ,  3.15718143e-03  , 1.02069266e-02  , 1.09132733e-01,
    8.73479113e-01],
 [  5.34282996e-03  , 4.23784292e-03  , 1.66599629e-02  , 1.14030772e-01,
    8.59728592e-01]]

thresh = [.005, .015, .03, .05, 0.9]

pred_classes = sequential_threshold(some_probs, thresh)

print(pred_classes)
