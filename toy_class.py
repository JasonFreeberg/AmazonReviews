#
#
#

class Base(object):

    def __init__(self):
        pass

    def transform(self, string):
        pass


class SpaceSplit(Base):

    def transform(self, string):
        return string.split(' ')

class UnderScoreSplit(Base):

    def transform(self, string):
        return string.split('_')


transform1 = SpaceSplit()
transform2 = UnderScoreSplit()

a = transform1.transform("My name is Jason_Freeberg!")
b = transform2.transform("My name is Jason_Freeberg!")

print(a)
print(b)
