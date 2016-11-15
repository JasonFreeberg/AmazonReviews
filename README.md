# The Language of Amazon Reviews
__Jason Freeberg__
__November 2016__

## Project Motivation

In 2015 alone, Amazon shipped over one billion packages on behalf of its sellers. And in the days leading to Christmas, Amazon was processing 426 orders per second. Obviously, Amazon is an important player in the online consumer and shipping industry. But my project does not focus on the company's massive throughput. Instead, I wanted to investigate the correlation between a user's text review and the overall rating they assigned to that product. 

When a user makes a purchase on Amazon, he or she takes time to decide between similar products. And any savvy shopper would likely browse through the product reviews. And while that shopper cannot read through every review, he or she can see the general sentiment of previous customers by the numerical rating. There should (ideally) be a strong correlation between the language used and the rating given, so this was a great opportunity to familiarize myself with Python's Natural Language Toolkit, or *nltk*. 

My question was **how well can we predict a user's product rating from their text review?** Moreover... what relevant features can be extracted from this data, and what models work best for prediction?

## The Data

In my search for data, I happened across [Julian McAley](http://jmcauley.ucsd.edu/data/amazon/) his website. There I found he had collected massive amounts of Amazon user reviews, each review saved as a JSON. I chose to investiage Julian's collection of 1.8 million electronic product reviews.  

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

An example of a single review's data.

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

For this project, I imported Julian's dataset into a local MongoDB instance on my laptop. I ~~used R to create a Shiny application for an interactive~~ made a Jupyer notebook for exploratory analysis. I used Python and nltk to extract features from the text, and fit a Naive Bayes classifier from sci-kit learn.

## Methodology

Once the data was queried from MongoDB, the data was split into k cross folds for model validation. For each train and test set pairs the script will aggregate the most common words and phrases. Using that data, each observation is mapped onto that *feature space* with other predictors like character count. The work flow of the project is outlined below.

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
