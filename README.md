# The Language of Amazon Reviews
__November 2016__

## Project Motivation

In 2015 alone, Amazon shipped over one billion packages on behalf of its sellers. And in the days leading to Christmas, Amazon processed 426 orders per second. Obviously, the company is an important player in the online consumer and shipping industry. While the company's massive throughput is very interesting, my project focuses on something every Amazon user can relate to... product reviews! Specifically, I wanted to investigate the correlation between a user's text review and the overall rating they assigned to that product. 

Before a user makes a purchase on Amazon, he or she takes some time to decide between similar products. And any savvy shopper would likely browse through each product's reviews. And while that shopper cannot read through every user review, he or she can see the *general sentiment* of previous customers by the number of stars assigned. There should be a strong correlation between the language used and the numerical rating given, so this was a great opportunity to familiarize myself with the basics of natural language processesing. 

Put simply, my question was **how well can we predict a user's product rating from their text review?** Moreover... what relevant features can be extracted from this data, and what models work best for prediction?

## The Data

In my search for data, I happened across [Julian McAley](http://jmcauley.ucsd.edu/data/amazon/) and his website. There I found he had collected massive amounts of Amazon user reviews, each review saved as a JSON. I chose to investiage Julian's collection of 1.8 million electronic product reviews. Each JSON contained the following fields...

- **\_id**... &nbsp; An id number added by MongoDB
- **reviewerID**... &nbsp; The reviewer ID number
- **asin**... &nbsp; The product's ID number
- **reviewerName**... &nbsp; Name of the reviewer
- **helpful**... &nbsp; Helpfulness score of the review [upvotes, total]
- **reviewText**... &nbsp; Text of the user's review
- **overall**... &nbsp; The overall rating, out of five stars
- **summary**... &nbsp; Summary of the review
- **unixReviewTime**... &nbsp; Unix timestamp of the review
- **reviewTime**... &nbsp; Raw time of the review

An example of a single review's data....

    {
        "_id" : ObjectId("5813fe29668cc684c99e695a"),
        "reviewerID" : "AO94DHGC771SJ",
        "asin" : "0528881469",
        "reviewerName" : "amazdnu",
        "helpful" : [ 0, 0 ],
        "reviewText" : "We got this GPS for my husband who is an (OTR) over the road trucker.  Very Impressed with the 
            shipping time, it arrived a few days earlier than expected...  within a week of use however it 
            started freezing up... could of just been a glitch in that unit.  Worked great when it 
            worked!  Will work great for the normal person as well but does have the \"trucker\" option. 
            (the big truck routes - tells you when a scale is coming up ect...)  Love the bigger screen, 
            the ease of use, the ease of putting addresses into memory.  Nothing really bad to say about the unit with
             the exception of it freezing which is probably one in a million and that's just my luck.  I contacted the 
             seller and within minutes of my email I received a email back with instructions for an exchange! VERY 
             impressed all the way around!",
        "overall" : 5,
        "summary" : "Gotta have GPS!",
        "unixReviewTime" : 1370131200,
        "reviewTime" : "06 2, 2013"
    }

## Tools

For this project, I imported Julian's dataset into a local **MongoDB** instance on my laptop. I ~~used R to create a Shiny application for an interactive~~ made a Jupyer notebook for exploratory analysis. I used **Python** and [nltk](http://www.nltk.org) to extract features from the text, and fit a Naive Bayes classifier from [scikit-learn](http://scikit-learn.org/stable/).

## Methodology

Once the data was queried from MongoDB, the data was split into 4 folds for model validation. For each train'test set pair, I aggregated the most common words and phrases. Using that data, each observation is mapped onto that *feature space* with other predictors like character count. The work flow of the project is outlined below.

```
1: Query MongoDB
2: Create k-Fold cross validation iterator
FOR train and test sets in cross validation iterator:
    3: Using training data:
        a: Aggregate features from text
        b: Create feature space
            - Top N words
    4: Map train and test observations onto feature space
            - Add character count
    5: Classify test set observations
        a: Create table of predicted class probabilities and true labels
    6: Save validation results
        a: Append to a table outside of loop
END
7: Examine results
    a: Confusion matrix
    b: Precision and Recall
```

## Features

- Bag-of-words
- Character Count
- Number of sentences
- "Chunks"

... Will elaborate here soon ;)

