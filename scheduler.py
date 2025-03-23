import mysql.connector
import smtplib

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()

s.login("kkiranmai02@gmail.com","Sriya@0902")

conn = mysql.connector.connect(host="localhost", user="root", password="", database="test")

cursor = conn.cursor()

#FETCH ALL USER-EMAILS FROM DATABASE

user_emails = 'select email from users_data'

cursor.execute(user_emails)
records = cursor.fetchall()

print("No of records: ", cursor.rowcount)

email =[]
for record in records:
    email.append(record[0])

print("emails: "+email)

#FETCH ALL NAMES FROM DATABASE
user_emails = 'select name from users_data'

cursor.execute(user_emails)
records = cursor.fetchall()

print("No of records: ", cursor.rowcount)

name =[]
for record in records:
    name.append(record[0])

print("Names: "+name)

#get dates per week
from datetime import datetime, timedelta
now = datetime.now() #current date
start_date_target = now - timedelta(days=7) #weekly records
end_date = now.strftime('%Y-%m-%d')
start_date = start_date_target.strftime('%Y-%m-%d')


#FIND THE PREDICTIONS FROM TABLE FOR EVERY-USER 

negative = ['Stress/Anxiety', 'Substance Abuse/Addiction', 'Marriage',
       'Relationship Dissolution', 'Behavioral Change', 'Relationships',
       'Trauma/Grief/Loss', 'Depression', 'Parenting', 'Intimacy',
       'Workplace issues', 'Family Conflict', 'LGBTQ', 'Self-esteem',
       'Counseling Fundamentals', 'Anger Management',
       'Social Relationships', 'Sleep Improvement']

self_focus = ['Stress/Anxiety','Trauma/Grief/Loss', 'Depression',  
              'Intimacy','Self-esteem','Anger Management', 'Sleep Improvement']

relations = ['Marriage',  'Relationship Dissolution', 'Relationships',
              'Social Relationships','Family Conflict', 'Parenting' ]

behaviour = ['Substance Abuse/Addiction','Behavioral Change', 'LGBTQ', 'Counseling Fundamentals']

workplace = ['Workplace issues']

alarming = ['Suicide']

#Mail-template fetch
mail_template_qry = "select DATA from system_prop where CODE = 'MAILTEMPLATE'"
cursor.execute(mail_template_qry)
record = cursor.fetchone()
mail_template = record[0]

idx = 0
for user in email:
    pred_his = "select prediction from pred_data where email = %s and date between %s and %s",(user, start_date, end_date,)
    cursor.execute(user_emails)
    records = cursor.fetchall()

    class_rep = []
    score = 0
    alert =0
    for record in records:
        instance = record[0]
        if instance in negative:
            score -=1
            if instance in self_focus:
                class_rep.append("self_focus")
            elif instance in relations:
                class_rep.append("relationships")
            elif instance in workplace:
                class_rep.append("workplace")
            else:
                class_rep.append("behaviour")

        elif instance in alarming:
            score -=2
            alert+=1
        else:
            score+=1
    
    if alert>1:
        final = 'alarming'
    elif score>0:
        final = 'positive'
    elif score==0:
        final = 'neutral'
    else:
        final = 'negative'
        ans =  max(set(class_rep), key = class_rep.count)

    
    #send mail to the user
    name_usr = name.index(idx)
    mail_template =  mail_template.replace("#callName#", name_usr)
    if final !='negative':
        mail_template = mail_template.replace("#mood#", final)
        mail_template = mail_template.replace("#resouces", resource_pos)
    if final == 'negative':
        mail_template = mail_template.replace("#mood#", ans)
        mail_template = mail_template.replace("#resources", resouce_neg)

    message = mail_template
    s.sendmail("kkiranmai02@gmail.com",user, message)

s.quit()
    

    

    
        






