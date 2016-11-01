# PSTAT 131 Project
Jason Freeberg
Fall 2016

A deep dive into Amazon reviews of electronic devices. Data courtesy of [Julian McAley](http://jmcauley.ucsd.edu/data/amazon/) from UC San Diego.

Variables in data:
- **\_id** is an id number added by MongoDB
- **reviewerID** The reviewer ID number
- **asin** The product ID number
- **reviewerName** Name of the reviewer,
- **helpful** Helpfulness rating of the review [upvotes, total]
- **reviewText** Text of the user's review'
- **overall** the overall rating. Out of five stars
- **summary** Summary of the review
- **unixReviewTime** Unix timestamp of the review
- **reviewTime** Raw time of the review

Example of the JSON:

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

... more to come as the quarter progresses. ;)
