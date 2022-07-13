from flask import Flask, make_response, render_template, request
import io
from io import StringIO
import csv
from graphviz import render
import pandas as pd
import numpy as np
import pickle


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
lemmatizer = WordNetLemmatizer()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return render_template('upload_csv.html')

@app.route('/transform', methods=["GET", "POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    data = pd.read_csv(stream)
    #data = list(csv.reader(stream))
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    
    print(type(data))
    
    stop_words = set(stopwords.words('english'))
    
    
    
    clean_text =[]
    for review in data['Text']:
        review= re.sub(r'[^\w\s]', '', str(review))
        # \s-->includes [ \t\n\r\f\v]
        # \w-->includes only [a-zA-Z0-9_]
        # removes "[","]","^"
        
        review = re.sub(r'\d','',review)
        #include int type [0-9]
        
        review_token = word_tokenize(review.lower().strip()) 
        #convert into lower case and strip leading and tailing spaces followed by spliting sentnece into words

        review_without_stopwords=[]
        
        # Condition to check, If it is not stopword we lemmatize and append into a list
        for token in review_token:
            if token not in stop_words:
                token= lemmatizer.lemmatize(token)
                review_without_stopwords.append(token)
        
        # and added to cleaned_review list 
        cleaned_review = " ".join(review_without_stopwords)
        clean_text.append(cleaned_review)

    

    stop = stopwords.words('english')
    text = clean_text
    text_tokens = word_tokenize(str(text))
    tokens_without_sw = [word for word in text_tokens if not word in stop]
    tokens_without_sw=str(tokens_without_sw)
    
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentence = tokens_without_sw

    tokenized_sentence = nltk.word_tokenize(sentence)

    sid = SentimentIntensityAnalyzer()
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]

    for word in tokenized_sentence:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:        
            neu_word_list.append(word) 



    Text = clean_text
    df = pd.DataFrame()
    df['Text']=Text

    sid = SentimentIntensityAnalyzer()
    #for sentence in Text:
        #print(sentence)
            
        #ss = sid.polarity_scores(sentence)
        #for k in ss:
            #print('{0}: {1}, ' .format(k, ss[k]), end='')
        #print()
            
    analyzer = SentimentIntensityAnalyzer()
    df['rating'] = df['Text'].apply(analyzer.polarity_scores)
    df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)
    ### Creating a dataframe.
    sorted_good_reviews=df.sort_values(by='compound', ascending=False)
    sentences=df

    #Assigning score categories and logic
    i = 0

    predicted_value = [ ] #empty series to hold our predicted values

    while(i<len(sentences)):
        if ((sentences.iloc[i]['compound'] >= 0.4)):
            predicted_value.append('positive')
            i = i+1
        elif ((sentences.iloc[i]['compound'] >= 0) & (sentences.iloc[i]['compound'] < 0.5)):
            predicted_value.append('neutral')
            i = i+1
        elif ((sentences.iloc[i]['compound'] < 0)):
            predicted_value.append('negative')
            i = i+1
    ## The threshold value will categorize if a given sentence is positive negative or neutral in nature.    

    df['Target'] = predicted_value 
    ## A new column has been created called as 'Target' with sentiments assigned to a given text.
    
    Prediction_df=df
    #used for prediction in last segment

    
    df.drop(['neg','neu','pos','compound'],axis=1,inplace=True)
    ## Dropping the neg, neu, pos, and compound columns.

    df['Star']=data['Star']

    count=df['Text'].count()
    count.dtype


    low_star_reviews = df[df.Star <3 ]
    avg_star_reviews = df[df.Star ==3]
    high_star_reviews =df[df.Star >3]

    low_star_reviews_with_pos_comments = low_star_reviews[low_star_reviews.Target != 'negative']
    df1=low_star_reviews_with_pos_comments

    avg_star_reviews_with_posorneg_comments = avg_star_reviews[avg_star_reviews.Target != 'neutral']
    df2=avg_star_reviews_with_posorneg_comments

    high_star_reviews_with_neg_comments=high_star_reviews[high_star_reviews.Target != 'positive']
    df3=high_star_reviews_with_neg_comments

    dataframe1 = pd.DataFrame(df1)
    dataframe2 = pd.DataFrame(df2)
    dataframe3 = pd.DataFrame(df3)
    merged_df = pd.concat([dataframe1, dataframe2,dataframe3])

    merged_df=merged_df.sort_values(by="Target", ascending=False)
    
    
    my_new_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
    


    tfidf =TfidfVectorizer()
    X = tfidf.fit_transform(Prediction_df['Text'].values.astype('U'))
    Y = Prediction_df['Target']
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)    

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve, recall_score
    from sklearn import metrics
    DT = DecisionTreeClassifier().fit(X,Y)

    #predict on train 
    train_preds2 = DT.predict(X_train)
    #accuracy on train
    #print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds2))
    l1 = accuracy_score(Y_train, train_preds2).round(3)
    
    #predict on test
    test_preds2 = DT.predict(X_test)
    #accuracy on test
    #print("Model accuracy on test is: ", accuracy_score(Y_test, test_preds2))
    l2 = accuracy_score(Y_test, test_preds2).round(3)

    #Confusion matrix
    #print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds2))
    l3 = confusion_matrix(Y_train, train_preds2).round(3)
    #print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds2))
    l4 = confusion_matrix(Y_test, test_preds2).round(3)
    # print('Wrong predictions out of total')
    

    # Wrong Predictions made.
    #print((Y_test !=test_preds2).sum(),'/',((Y_test == test_preds2).sum()+(Y_test != test_preds2).sum()))
    l5 = (Y_test !=test_preds2).sum()
    l6 = ((Y_test == test_preds2).sum()+(Y_test != test_preds2).sum())
    

    # Kappa Score
    #print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds2))
    l7 = metrics.cohen_kappa_score(Y_test,test_preds2).round(3)

    #stream.seek(0)

    #opt_df = my_new_df[:].values


    return render_template('results.html', data_var=my_new_df, l1=l1,l2=l2,l3=l3,l4=l4,l5=l5,l6=l6,l7=l7)
    #return render_template('results.html', table = opt_df, l1=l1,l2=l2,l3=l3,l4=l4,l5=l5,l6=l6,l7=l7)
    

if __name__ == "__main__":
    app.run(debug=False,port=9000)