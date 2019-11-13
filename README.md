# Titanic: Machine learning from the disaster
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

#### Data 
Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition.

This dataset contains information about 891 people who were on board the ship when departed on April 15th, 1912. As noted in the description on Kaggle's website, some people aboard the ship were more likely to survive the wreck than others. There were not enough lifeboats for everybody so women, children, and the upper-class were prioritized. Using the information about these 891 passengers, try to build a model to predict which people would survive based on the following fields:

- **Name** (str) - Name of the passenger
- **Pclass** (int) - Ticket class
- **Sex** (str) - Sex of the passenger
- **Age** (float) - Age in years
- **SibSp** (int) - Number of siblings and spouses aboard
- **Parch** (int) - Number of parents and children aboard
- **Ticket** (str) - Ticket number
- **Fare** (float) - Passenger fare
- **Cabin** (str) - Cabin number
- **Embarked** (str) - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## EDA: Exploratory Data Analysis
Tool: Python(numpy, pandas, matpotlib.pyplot, seaborn)

Objective: Finding reasons and trends for which people were more likely to survive the wreck than others.
 
#### Plot continuous features

Histogram for people who survived or did not survived based on their Age
<img width="367" alt="Screen Shot 2019-11-13 at 4 39 48 PM" src="https://user-images.githubusercontent.com/43712046/68810623-40982400-0634-11ea-806f-7f9c1d85216e.png">

Histogram for people who survived or did not survived based on the fare price they paid
<img width="367" alt="Screen Shot 2019-11-13 at 4 40 27 PM" src="https://user-images.githubusercontent.com/43712046/68810641-4f7ed680-0634-11ea-9bfa-ba3ab6dbbc7f.png">

Categorical plot ofpeople who survived based on the ticket class
<img width="500" alt="Screen Shot 2019-11-13 at 4 44 18 PM" src="https://user-images.githubusercontent.com/43712046/68810877-f2cfeb80-0634-11ea-80fe-0372e3bba2d0.png">

**It can be clearly seen that people with ticket class 1 were more likely to survive than 2 or 3**

Combining the SibSp and Parch and, plotting a Catplot of people who survived based on the Family Count
<img width="500" alt="Screen Shot 2019-11-13 at 4 49 14 PM" src="https://user-images.githubusercontent.com/43712046/68811138-873a4e00-0635-11ea-8773-189b305f019a.png">

**It can be incurred that the lesser the family size the more likely to survive**

#### Plot categorical features

<img width="300" alt="Screen Shot 2019-11-13 at 4 55 54 PM" src="https://user-images.githubusercontent.com/43712046/68811515-7807d000-0636-11ea-8641-e38746cc83aa.png"><img width="300" alt="Screen Shot 2019-11-13 at 4 56 20 PM" src="https://user-images.githubusercontent.com/43712046/68811537-85bd5580-0636-11ea-9f90-296cd5bc448a.png"><img width="300" alt="Screen Shot 2019-11-13 at 4 56 45 PM" src="https://user-images.githubusercontent.com/43712046/68811572-95d53500-0636-11ea-9a05-5d4e52c87abb.png">
