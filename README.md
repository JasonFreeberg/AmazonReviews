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

For this project, I imported Julian's dataset into a local **MongoDB** instance on my laptop. I made a Jupyter notebook for exploratory analysis. I decided to use **Python** and [Sci Kit's](http://scikit-learn.org/stable/) Term-Frequency Inverse Document Frequency (tfidf) class to extract features from the text. I then compared a Support Vector Machine and Naive Bayes Classifier. 

## Methodology

Once the data was queried from Mongo, I split it into a train and test set. In the test set, I used Sci Kit's Pipeline and GridSearchCV classes to pipe together the tfidf transformer and classifiers. The GridSearchCV class allowed me to perform a grid search over the hyper parameters of both the transformers and classifiers. The process is time consuming.

```
1: Query MongoDB
2: Split into testing and training sets
3: Create dev-train and dev-test sets from training set
4: Create 2 classification pipelines
    - tfidf transformer -> PCA -> classifier

FOR each pipeline:
    FOR each combination of hyperparemeters in GridSearchCV:
        - Score classifier and average over two folds

5: Examine results
    a: Confusion matrix
    b: Precision and Recall or AUC
```

## Results
