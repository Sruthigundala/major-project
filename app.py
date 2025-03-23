from flask import Flask, render_template, url_for, request,session, redirect
import mysql.connector, re, pickle

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer, CountVectorizer

#STOP WORDS, STEMMING & LEMMATIZING

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import FreqDist, pos_tag

import re
import scipy
from sentence_transformers import SentenceTransformer

from joblib import load

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('omw-1.4')


#stop words

stop_words = list(stopwords.words('english'))
stop_words

#adding few more words to suit functionality
stop_words.extend(["ex","husband","wife","girl","boy","n't",'thank','get','please','san','diego',
                   'hi','hello','im','wa','ha','ive','would','like','know','also','let',
                   '2020','name','one','yet','june','said','two','aug','oct','jan','dec','july',
                   '-','january','april','robert','etc','nov','mesa','brown','andrew','jean','kim'])

def get_wordnet_pos(word):
   tag = pos_tag([word])[0][1][0].upper()
   #print(tag)
   tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
   return tag_dict.get(tag, wordnet.NOUN)


def similarity_test(query):
    model=SentenceTransformer('bert-base-nli-mean-tokens')
    input_msg=[query]
    input_embeddings= model.encode(input_msg)
    queries=['How can I control myself and my anger?',
       'How do I not be angry all the time?',
       'How can I control my temper?', 'Why do I get angry so easily?',
       'How do I deal with my alcoholic boyfriend with a dark past?',
       'My girlfriend always brings up past events and talks negatively about them',
       'Why did my boyfriend hit himself in the face during an argument?',
       'How can I just be happy and not mad all the time?',
       'How can I be less angry?', 'I need help controlling my anger',
       "How do I deal with my son's violent thoughts and dreams?",
       "How can I deal with the anger problems I've gained from my soon-to-be husband?",
       'How can I stop being so angry?', 'Why am I so mad?',
       'How can I control myself and learn to let things go or communicate?',
       'How can I be less confused about my feelings towards anything?',
       'I need to know how to cope with misophonia before I go completely insane',
       'i need answers to my angry, possessiveness, and urges',
       'How can I control my anger?', 'How do I manage my anger?',
       'Why am I constantly angry?',
       'My husband and I canâ€™t talk to each other without arguing',
       'Why do I get random spurts of anger over petty things?',
       'I have anger issues. I am extremely explosive about the simplest things']
    query_embeddings=model.encode(queries)
    closest_n=1
    dist_scores=[]
    for query,query_embedding in zip(queries,query_embeddings):
        distances=scipy.spatial.distance.cdist([query_embedding],input_embeddings,"cosine")[0]
        score=(1-distances)
        dist_scores.extend(score)
    return max(dist_scores)


def Vader_sentiment_scores(sentence):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] > 0.2 :
        return("Positive")
    
  
    else :
        return("Negative")  


def lemmatize(word):
    wnl = WordNetLemmatizer()
    word = wnl.lemmatize(word, pos='v')
    word = wnl.lemmatize(word, pos='n')
    word = wnl.lemmatize(word, pos='a')
    return word


def cleanText(text):
    # emojis pattern source is stackoverflow
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        re.UNICODE)

    # Removing links, non letters, eemojis, and turning to lower case
    text = re.sub(r'http\S+ ', '', text).lower()
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    text = re.sub(emoj, "", text)

    # Lemmatizing and removing stop words
    text = list(map(lemmatize, text.split()))
    text = ' '.join([word for word in text if word not in stop_words])

    return text

flag =0
def classifyText(text):
    text = cleanText(text)
    text = cv.transform([text]).toarray()
    pred = model.predict(text)
    if pred:
        flag = 1
        print('The text is suicidal')
    else:
        flag = 0
        print('The text is non suicidal')
    return flag


def chatbot_response(msg):
    msg=msg.lower()
    ans = classifyText(msg)
    if ans ==1:
        answer='Pls contact helpline!'
    else:
        sentiment=Vader_sentiment_scores(msg)
        greetings = ['hi','hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you', "i'm doing well", 'pleasure to meet you', 'hi whatsup','Hi!', 'Hi !', 'hi !', 'hi!', 'hola!', 'Hola !','Hola!']
        goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later', 'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night', 'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then', 'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day', 'thanks a bunch', "i can't thank you enough", "you're great", 'thanks a ton', 'grateful for your help', 'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']
        if msg in greetings:
            answer="Hi! I\'m a Mental Health chatbot!.You can call me Buddy. Please ask me for help whenever you feel like it! I\'m always online."
        elif msg in goodbyes:
            answer="Hope I was able to help you today! Take care, bye!"
        elif sentiment=="Positive":
            answer="That's great! Do you still have any questions for me?"
        else:
            #preprocess the incoming text
            text = msg
            text = str(text).lower()#Convert into lower case
            #remove all charac except alphabets
            text = re.sub('^[a-zA-Z]',' ',text)
            #remove punctuation
            text= re.sub('[^\w\s]', '', text)
            text = re.sub('\d+', '',text)
            text = re.sub(',', ' ', text)


            #Tokenizing &Lemmatization

            lemmatizer = WordNetLemmatizer()
            lem_words = []
            tokens = word_tokenize(text)

            for token in tokens:
                if token not in stop_words:
                    #print(token)
                    #print(get_wordnet_pos(token))
                    lemmetized_word = lemmatizer.lemmatize(token, get_wordnet_pos(token))
                    lem_words.append(lemmetized_word)

            #REMOVE LETTERS IF ANY AFTER CLEANING 
            lem_words = [i for i in lem_words if len(i)>2]
            text = ' '.join(lem_words)

            #pickle files
            Count_Vect_ = pickle.load(open("count_vect_svm.pkl", "rb"))
            Transformer_ = pickle.load(open("transformer_svm.pkl", "rb"))
            SVM_tf = pickle.load(open("model_svm.pkl", "rb"))

            #Predict probabilities with TF-IDF(SVM), RF
            counts = Count_Vect_.transform([text])
            counts = Transformer_.transform(counts)
            tfidf_svm= SVM_tf.predict_proba(counts)

            #Classification
            classes=['Anger Management' ,'Behavioral Change', 'Counseling Fundamentals',
                    'Depression' ,'Family Conflict' ,'Intimacy', 'LGBTQ',
                    'Marriage', 'Parenting', 'Relationship Dissolution', 'Relationships',
                    'Self-esteem' ,'Sleep Improvement' ,'Social Relationships' ,'Stress/Anxiety',
                    'Substance Abuse/Addiction' ,'Trauma/Grief/Loss', 'Workplace issues']
            
            #Prediction of label
            maxpos = tfidf_svm.argmax()
            label=classes[maxpos]
            print("Label predicted: "+label)
            
            category = label


            #category=Classify_problem(msg)
            df=pd.read_excel("new-Answers.xlsx")
            df = df[df['topic'] == category]
            query=df['questionTitle']
            query = query.to_list()

            print("Query: " , query)
            scores=[]
            for queries in query:
                a=similarity_test(queries)
                scores.append(a)

            df['scores']=scores
            df=df[df['scores']==df['scores'].max()]
            answer_out=df['answerText'].tolist()
            answer_out.insert(0, "Tough to hear that.Here is my solution")
            answer = ''.join(answer_out)
    return answer
    
#load the files
cv = load('cv.joblib')
model = load('model.joblib')

app = Flask(__name__)

conn = mysql.connector.connect(host="localhost", user="root", password="", database="test")
cursor = conn.cursor()
app.secret_key='secret'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', username= session['username'])

@app.route('/info')
def info():
    return render_template('project-info.html')

@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute('SELECT * FROM users_data WHERE email = %s AND password = %s', (username, password, ))
        account = cursor.fetchone()
        recordNo = cursor.rowcount
        if recordNo!=0:
            session['loggedin'] = True
            session['username'] = account[1]
            session['email'] = account[4]
            msg = 'Logged in successfully !'
            return redirect(url_for('about'))
        else:
            msg = 'Incorrect username / password !'
            return render_template('login.html', msg = msg)
    else:
        return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('email', None)
    return redirect(url_for('login'))
    
@app.route('/burnout', methods=['GET', 'POST'])
def burnout():
    msg =''
    ans =''
    if request.method =='POST':
        eval = request.form['eval']
        pcount = request.form['pcount']
        avghrs = request.form['avghrs']
        workingYrs = request.form['workingYrs']
        accidents = request.form['accidents']
        turnover = request.form['turnover']
        promotion = request.form['promotion']
        salary = request.form['salary']
        dept = request.form['dept']

        emp = pickle.load(open("emp_burnout.pkl", "rb"))

        data = [[eval, pcount, avghrs, workingYrs, accidents, turnover, promotion, salary, dept]]
        pred = emp.predict(data)

        print(pred)

        if pred<=0.25:
            ans = '0'
            msg = 'Least Satisfied'
        elif pred>0.25 and pred<=0.50:
            ans = '1'
            msg = 'Moderately Satisfied'
        elif pred>0.50 and pred<=0.75:
            ans = '2'
            msg = 'Satisfied well'
        else :
            ans ='3'
            msg ='Extremely Satisfied'
        
        return render_template('employeeBurnout.html', msg=msg, ans=ans)


    return render_template("employeeBurnout.html", msg=msg, ans=ans)


@app.route('/resource')
def resource():
    return render_template('resource.html')

@app.route('/journal', methods= ['GET', 'POST'])
def journal():

    msg = ''
    if request.method == 'POST':
        text = original = request.form['entry']

        from datetime import datetime
        now = datetime.now()
        formatted_date = now.strftime('%Y-%m-%d')

        #suicide detection
        ans = classifyText(text)
        if ans==1:
            msg = 'Suicide'

        if ans!=1:
            #preprocess the incoming text
            text = str(text).lower()#Convert into lower case
            #remove all charac except alphabets
            text = re.sub('^[a-zA-Z]',' ',text)
            #remove punctuation
            text= re.sub('[^\w\s]', '', text)
            text = re.sub('\d+', '',text)
            text = re.sub(',', ' ', text)


            #Tokenizing &Lemmatization

            lemmatizer = WordNetLemmatizer()
            lem_words = []
            tokens = word_tokenize(text)

            for token in tokens:
                if token not in stop_words:
                    #print(token)
                    #print(get_wordnet_pos(token))
                    lemmetized_word = lemmatizer.lemmatize(token, get_wordnet_pos(token))
                    lem_words.append(lemmetized_word)

            #REMOVE LETTERS IF ANY AFTER CLEANING 
            lem_words = [i for i in lem_words if len(i)>2]
            text = ' '.join(lem_words)

            #pickle files
            Count_Vect_ = pickle.load(open("count_vect_svm.pkl", "rb"))
            Transformer_ = pickle.load(open("transformer_svm.pkl", "rb"))
            SVM_tf = pickle.load(open("model_svm.pkl", "rb"))

            #Predict probabilities with TF-IDF(SVM), RF
            counts = Count_Vect_.transform([text])
            counts = Transformer_.transform(counts)
            tfidf_svm= SVM_tf.predict_proba(counts)

            #Vader sentiment Analysis for predicting positive Sentences.
            vad_pred = Vader_sentiment_scores(text)
            print("Vader predicted: "+ vad_pred)

            msg = vad_pred

            if not vad_pred == "Positive":
                #Classification
                classes=['Anger Management' ,'Behavioral Change', 'Counseling Fundamentals',
                        'Depression' ,'Family Conflict' ,'Intimacy', 'LGBTQ',
                        'Marriage', 'Parenting', 'Relationship Dissolution', 'Relationships',
                        'Self-esteem' ,'Sleep Improvement' ,'Social Relationships' ,'Stress/Anxiety',
                        'Substance Abuse/Addiction' ,'Trauma/Grief/Loss', 'Workplace issues']
            
                #Prediction of label
                maxpos = tfidf_svm.argmax()
                label=classes[maxpos]
                print("Label predicted: "+label)
            
                msg = label

        email = session['email']
        cursor.execute('INSERT INTO pred_data VALUES (NULL, %s, %s, %s, %s)', (email, formatted_date, original, msg, ))
        conn.commit()

        return render_template('journal.html', msg = 'Entry Recorded')
    
    return render_template('journal.html', msg = msg)    


@app.route('/personality', methods= ['GET', 'POST'])
def personality():
    msg = ''
    if request.method=='POST':
        print('YES')
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']
        q6 = request.form['q6']
        q7 = request.form['q7']
        q8 = request.form['q8']
        q9 = request.form['q9']
        q10 = request.form['q10']
        q11 = request.form['q11']
        q12 = request.form['q12']
        q13 = request.form['q13']
        q14 = request.form['q14']
        q15 = request.form['q15']
        q16 = request.form['q16']
        q17 = request.form['q17']
        q18 = request.form['q18']
        q19 = request.form['q19']
        q20 = request.form['q20']

        questions = []
        questions.extend([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20] )

        count_a =0
        count_b =0
        count = 0

        personality_dichotomy = ''

        for ques in questions:
            if ques=="1":
                count_a+=1
            elif ques=="2":
                count_b+=1
            count +=1

            if count==5:
                if count_a > count_b:
                    personality_dichotomy = personality_dichotomy + 'E '
                else:
                    personality_dichotomy = personality_dichotomy + 'I '
            
            if count == 10:
                if count_a > count_b:
                    personality_dichotomy = personality_dichotomy + 'S '
                else:
                    personality_dichotomy = personality_dichotomy + 'N '
            
            if count == 15:
                if count_a > count_b:
                    personality_dichotomy = personality_dichotomy + 'T '
                else:
                    personality_dichotomy = personality_dichotomy + 'F '
        
            if count == 20:
                if count_a > count_b:
                    personality_dichotomy = personality_dichotomy + 'J '
                else:
                    personality_dichotomy = personality_dichotomy + 'P '

        msg = personality_dichotomy
        return render_template('personality.html', msg = msg)

    return render_template('personality.html', msg = msg)

@app.route('/userRecords')
def userRecords():
    email = session['email']
    name = session['username']
    cursor.execute('Select * from pred_data where email = %s ORDER BY date DESC ',(email,))
    value = cursor.fetchall()
    return render_template('userRecords.html', data=value, name=name)

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['pwd']
        conf_password = request.form['conf_pwd']
        email = request.form['email']
        age = request.form['age']
        gender = request.form['gender']
        

        cursor.execute('SELECT * FROM users_data WHERE email = %s ', (email, ))
        account = cursor.fetchone()
        
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z]+', username):
            msg = 'Name must contain only characters without spaces !'
        elif not re.match(r'[0-9]+', age):
            msg = 'Age Must have Numeric Input!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        
        elif len(password)<8:
            msg = 'Password must contain atleast 8 characters !'
        elif not re.match(r'^(?=.*[a-z])', password):
            msg = 'Password must contain atleast one lowercase letter !'
        elif not re.match(r'^(?=.*[A-Z])', password):
            msg = 'Password must contain atleast one uppercase letter !'
        elif not re.match(r'^(?=.*?[0-9])', password):
            msg = 'Password must contain atleast one digit !'
        elif not re.match(r'^(?=.*?[#?!@$%^&*-])', password):
            msg = 'Password must contain atleast one special character !'

        elif not conf_password == password:
            msg = 'Confirm Password must match the original Password'
        
        else:
            
            cursor.execute('INSERT INTO users_data VALUES (NULL, %s, %s, %s, %s, %s)', (username, age, gender, email, password, ))
            conn.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

@app.route('/chatbot', methods =['GET', 'POST'])
def chatbot():
    return render_template('chatbot.html')

@app.route('/getResponse', methods =['GET','POST'])
def getResponse():
    text = request.args.get('msg')
    ans = chatbot_response(text)
    return ans


if __name__ == '__main__':
    app.run(debug = True)
