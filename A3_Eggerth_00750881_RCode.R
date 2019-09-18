# AUFGABENBLATT 3
# Cordula Eggerth (00750881)

# Verwendete Literaturquellen:
# Folien und R-Codes zu den bisher vorgetragenen Kapiteln aus UK Erweiterungen des linearen Modells (Prof. Marcus Hudec).
# Weitere Quellen:
# https://stats.stackexchange.com/questions/49416/decision-tree-model-evaluation-for-training-set-vs-testing-set-in-r

rm(list=ls())

install.packages("kernlab")
install.packages("rpart")
install.packages("verification")
install.packages("ROCR")
install.packages("pROC")
install.packages("Deducer")

library(kernlab)
library(ggplot2)
library("MASS")
library("rpart")
library("verification")
library(class)
library(ROCR)
library(pROC)
library(Deducer)

setwd("C:/Users/Coala/Desktop/A3_ERWEIT")

#***********************************************************************************************
# AUFGABE 1
#***********************************************************************************************
# Erweitern Sie die in der Vorlesung vorgestellten Analysen zum SPAM-Dataset durch die 
# Verwendung möglichst aller Prädiktoren.
# Wenden Sie sowohl die Methoden Classification Tree als auch Logistische Regression an.
# Zeigen Sie die Trennschärfe Ihres Modells mit der ROC-Kurve.
# Quelle fuer Daten: https://rdrr.io/cran/kernlab/man/spam.html.

# deskriptive statistiken
data(spam)
attach(spam)
head(spam, n=5)
nrow(spam)
ncol(spam)
summary(spam)
cols <- colnames(spam)
cols2 <- paste(cols, collapse="+")

sum(spam$type=="nonspam")
sum(spam$type=="spam")

summary(spam[spam$type=="nonspam",])
summary(spam[spam$type=="spam",])

par(mfrow=c(2,2))
boxplot(spam$direct ~ spam$type, main="direct")
boxplot(spam$free ~ spam$type, main="free")
boxplot(spam$num000 ~ spam$type, main="num000")
boxplot(spam$credit ~ spam$type, main="credit")

# classification tree: recursive partitioning (rpart)
factor.type <- factor(spam$type)
is.factor(factor.type)

 # FALL 1: mit wahl von cp=0.0001 (ergibt großen tree mit vielen splits)
spam.rp1 <- rpart(factor.type ~ make+address+all+num3d+our+over+remove+
                      internet+order+mail+receive+will+people+
                      report+addresses+free+business+email+you+
                      credit+your+font+num000+money+hp+hpl+george+
                      num650+lab+labs+telnet+num857+data+num415+
                      num85+technology+num1999+parts+pm+direct+cs+
                      meeting+original+project+re+edu+table+conference+
                      charSemicolon+charRoundbracket+charSquarebracket+
                      charExclamation+charDollar+charHash+capitalAve+
                      capitalLong+capitalTotal, cp=0.0001)

summary(spam.rp1)

x11()
par(mfrow=c(1,1))
plot(spam.rp1)
text(spam.rp1, cex=0.6)

printcp(spam.rp1)
plotcp(spam.rp1)


# FALL 2: mit wahl von cp=0.003 (ergibt kleineren tree mit weniger splits)
spam.rp2 <- rpart(factor.type ~ make+address+all+num3d+our+over+remove+
                      internet+order+mail+receive+will+people+
                      report+addresses+free+business+email+you+
                      credit+your+font+num000+money+hp+hpl+george+
                      num650+lab+labs+telnet+num857+data+num415+
                      num85+technology+num1999+parts+pm+direct+cs+
                      meeting+original+project+re+edu+table+conference+
                      charSemicolon+charRoundbracket+charSquarebracket+
                      charExclamation+charDollar+charHash+capitalAve+
                      capitalLong+capitalTotal, cp=0.003)

summary(spam.rp2)

x11()
par(mfrow=c(1,1))
plot(spam.rp2)
text(spam.rp2, cex=0.6)

printcp(spam.rp2)
plotcp(spam.rp2)


# FALL 3: mit wahl von cp=0.007 (ergibt kleineren tree mit weniger splits)
spam.rp3 <- rpart(factor.type ~ make+address+all+num3d+our+over+remove+
                      internet+order+mail+receive+will+people+
                      report+addresses+free+business+email+you+
                      credit+your+font+num000+money+hp+hpl+george+
                      num650+lab+labs+telnet+num857+data+num415+
                      num85+technology+num1999+parts+pm+direct+cs+
                      meeting+original+project+re+edu+table+conference+
                      charSemicolon+charRoundbracket+charSquarebracket+
                      charExclamation+charDollar+charHash+capitalAve+
                      capitalLong+capitalTotal, cp=0.007)

summary(spam.rp3)

x11()
par(mfrow=c(1,1))
plot(spam.rp3)
text(spam.rp3, cex=0.6)

printcp(spam.rp3)
plotcp(spam.rp3)

# FALL 4: mit wahl von cp=0.05 (ergibt kleineren tree mit weniger splits)
spam.rp4 <- rpart(factor.type ~ make+address+all+num3d+our+over+remove+
                    internet+order+mail+receive+will+people+
                    report+addresses+free+business+email+you+
                    credit+your+font+num000+money+hp+hpl+george+
                    num650+lab+labs+telnet+num857+data+num415+
                    num85+technology+num1999+parts+pm+direct+cs+
                    meeting+original+project+re+edu+table+conference+
                    charSemicolon+charRoundbracket+charSquarebracket+
                    charExclamation+charDollar+charHash+capitalAve+
                    capitalLong+capitalTotal, cp=0.05)

summary(spam.rp4)

x11()
par(mfrow=c(1,1))
plot(spam.rp4)
text(spam.rp4, cex=0.6)

printcp(spam.rp4)
plotcp(spam.rp4)


# auto-pruning fuer fall 1:
spam.rp1.pruned <- prune(spam.rp1, cp=0.003)
x11()
plot(spam.rp1.pruned)
text(spam.rp1.pruned, cex=0.5)


# ROC-KURVEN, um trennschaerfe zu zeigen (fuer recursive partioning)
par(mfrow=c(2,2))

# ROC-kurve fuer fall 1: 
# predict:
pred <- prediction(predict(spam.rp1, type = "prob")[, 2], factor.type)
# plot ROC-kurve:
plot(performance(pred, "tpr", "fpr"), col="mediumorchid3", lwd=2,
     main="Fall 1: cp=0.0001")
abline(0, 1, lty = 2, col="dodgerblue2")

# ROC-kurve fuer fall 2: 
  # predict:
pred <- prediction(predict(spam.rp2, type = "prob")[, 2], factor.type)
  # plot ROC-kurve:
plot(performance(pred, "tpr", "fpr"), col="mediumorchid3", lwd=2, 
     main="Fall 2: cp=0.003")
abline(0, 1, lty = 2, col="dodgerblue2")

# ROC-kurve fuer fall 3: 
# predict:
pred <- prediction(predict(spam.rp3, type = "prob")[, 2], factor.type)
# plot ROC-kurve:
plot(performance(pred, "tpr", "fpr"), col="mediumorchid3", lwd=2, 
     main="Fall 3: cp=0.007")
abline(0, 1, lty = 2, col="dodgerblue2")

# ROC-kurve fuer fall 4: 
# predict:
pred <- prediction(predict(spam.rp4, type = "prob")[, 2], factor.type)
# plot ROC-kurve:
plot(performance(pred, "tpr", "fpr"), col="mediumorchid3", lwd=2, 
     main="Fall 4: cp=0.05")
abline(0, 1, lty = 2, col="dodgerblue2")



# logistische regression

res.logit <- glm(factor.type ~ make+address+all+num3d+our+over+remove+
                   internet+order+mail+receive+will+people+
                   report+addresses+free+business+email+you+
                   credit+your+font+num000+money+hp+hpl+george+
                   num650+lab+labs+telnet+num857+data+num415+
                   num85+technology+num1999+parts+pm+direct+cs+
                   meeting+original+project+re+edu+table+conference+
                   charSemicolon+charRoundbracket+charSquarebracket+
                   charExclamation+charDollar+charHash+capitalAve+
                   capitalLong+capitalTotal, 
                 family = binomial(link=logit))
res.logit
summary(res.logit)

res.probit <- glm(factor.type ~ make+address+all+num3d+our+over+remove+
                   internet+order+mail+receive+will+people+
                   report+addresses+free+business+email+you+
                   credit+your+font+num000+money+hp+hpl+george+
                   num650+lab+labs+telnet+num857+data+num415+
                   num85+technology+num1999+parts+pm+direct+cs+
                   meeting+original+project+re+edu+table+conference+
                   charSemicolon+charRoundbracket+charSquarebracket+
                   charExclamation+charDollar+charHash+capitalAve+
                   capitalLong+capitalTotal, 
                 family = binomial(link=probit))
res.probit
summary(res.probit)

# ROC-kurven fuer logistische regression (logit):

# methode 1:
par(mfrow=c(1,1))
prob.pred = predict(res.logit,type=c("response"))
spam$prob.pred=prob.pred
roc.1 <- roc(factor.type ~ prob.pred, data=spam)
plot(roc.1, col="lightslateblue") 

# methode 2:
modelfit <- glm(formula=factor.type ~ make+address+all+num3d+our+over+remove+
                  internet+order+mail+receive+will+people+
                  report+addresses+free+business+email+you+
                  credit+your+font+num000+money+hp+hpl+george+
                  num650+lab+labs+telnet+num857+data+num415+
                  num85+technology+num1999+parts+pm+direct+cs+
                  meeting+original+project+re+edu+table+conference+
                  charSemicolon+charRoundbracket+charSquarebracket+
                  charExclamation+charDollar+charHash+capitalAve+
                  capitalLong+capitalTotal, 
                family=binomial(), data=spam, na.action=na.omit)
rocplot(modelfit)



#***********************************************************************************************
# AUFGABE 2
#***********************************************************************************************
# Verwenden Sie den Datensatz http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#.
# Zur Absicherung gegenüber Over-Fitting ziehen Sie eine Zufallsstichprobe von 70% der Daten. 
# Entwickeln Sie damit einen Classification Tree und prüfen dessen Trennschärfe mittels der 
# ROC-Analyse für Training- und Test-Daten (70% - 30%).
# anmerkung: y ... yes/no (has the client subscribed a term deposit?)

# daten einlesen
bank.data <- read.csv("bank-full.csv", sep=";")
attach(bank.data)

# deskriptive statistiken
rows <- nrow(bank.data)
cols <- ncol(bank.data)
head(bank.data, n=5)

summary(bank.data[bank.data$y=="no",])
summary(bank.data[bank.data$y=="yes",])

par(mfrow=c(2,2))
boxplot(age ~ y, main="age")
boxplot(balance ~ y, main="balance")
boxplot(duration ~ y, main="duration")
boxplot(pdays ~ y, main="pdays")

# zufallsstichprobe ziehen (70% der daten)
training_size <- round(0.7*rows, digits=0)
random_indices <- sample(1:rows,training_size,replace=FALSE)
training.data <- bank.data[random_indices, ] 
test.data <- bank.data[-random_indices, ]
  
  # 1 ... training sample
  # 2 ... test sample
splitting <- rep(2,rows)
splitting[random_indices] <- 1 # 70% data is training sample

# classification tree (mit rpart)
is.factor(y) # y ist schon ein factor
cols <- colnames(bank.data)
cols2 <- paste(cols, collapse="+")

bank.rp <- rpart(y ~ age+job+marital+education+
                     default+balance+housing+loan+
                     contact+day+month+duration+
                     campaign+pdays+previous+poutcome, 
                 subset=splitting==1, 
                 parms=list(split="gini"),
                 cp=0.003)

summary(bank.rp)

x11()
par(mfrow=c(1,1))
plot(bank.rp)
text(bank.rp, cex=0.7)

printcp(bank.rp)
plotcp(bank.rp)

# ROC-kurve (pruefe trennschaerfe; mittels training (70%) - test (30%) split)

# within sample (for TRAINING data)
bank.probs.rpart <- predict(bank.rp, newdata=bank.data[splitting==1, ])
bank.preds.rpart <- predict(bank.rp, newdata=bank.data[splitting==1, ], 
                            type="class")
confusion(bank.preds.rpart, y[splitting==1])

par(mfrow=c(1,2))
roc.plot(y[splitting==1], bank.probs.rpart[,2], legend=TRUE, 
         leg.text="RPART",
         plot.thres=NULL, main="Training Sample")

par(mfrow=c(1,1))
library(caret)
library(ROCR)
roc_pred <- prediction(bank.probs.rpart[,2], training.data$y)
x11()
plot(performance(roc_pred, measure="tpr", x.measure="fpr"), colorize=TRUE)

# sensitivity / speficicity kurve  
plot(performance(roc_pred, measure="sens", x.measure="spec"), colorize=TRUE)

# out of sample test (for test data)                 
bank.probs.rpart <- predict(bank.rp, newdata=bank.data[splitting==2, ])
bank.preds.rpart <- predict(bank.rp, newdata=bank.data[splitting==2, ], 
                            type="class")
confusion(bank.preds.rpart, y[splitting==2])

x11()
roc.plot(as.numeric(as.character(y[splitting==2])), bank.probs.rpart[,2], 
         legend=TRUE, 
         leg.text="RPART",
         plot.thres=NULL, main="Test Sample")


par(mfrow=c(1,1))
library(caret)
library(ROCR)
roc_pred <- prediction(bank.probs.rpart[,2], test.data$y)
x11()
plot(performance(roc_pred, measure="tpr", x.measure="fpr"), colorize=TRUE)

# sensitivity / speficicity kurve  
plot(performance(roc_pred, measure="sens", x.measure="spec"), colorize=TRUE)



#***********************************************************************************************
# AUFGABE 3
#***********************************************************************************************
# Verwenden Sie den Datensatz https://stats.idre.ucla.edu/stat/data/binary.csv.
# Zielvariable: Admission to graduate school admit =1 / don't admit =0.
# Erklärende Variablen: GRE (Graduate Record Exam scores), GPA (grade point average) und 
#                       Prestige der Undergraduate Institution (Faktor mit Werten 1-4).
# Wenden Sie die Logistische Regression an und wählen Sie ein geeignetes Modell. 
# (Prüfen Sie auch auf etwaige relevante Interaktionseffekte.)
# Zeigen Sie die Trennschärfe Ihres Modells mit der ROC-Kurve.

# daten einlesen
grad.data <- read.csv("binary.csv", sep=",")
attach(grad.data)

# deskriptive statistiken
rows <- nrow(grad.data)
cols <- ncol(grad.data)
head(grad.data, n=10)

summary(grad.data[grad.data$admit==0,])
summary(grad.data[grad.data$admit==1,])

par(mfrow=c(2,2))
boxplot(gre ~ admit, main="gre")
boxplot(gpa ~ admit, main="gpa")
boxplot(rank ~ admit, main="rank")

factor.admit <- factor(admit)

# logistische regression & modellwahl
# modell 1: additives modell
res.logit.1 <- glm(factor.admit ~ gre + gpa + rank, 
                 family = binomial(link=logit))
res.logit.1
summary(res.logit.1)

# modell 2: mit interaktionen
res.logit.2 <- glm(factor.admit ~ gre*gpa*rank, 
                 family = binomial(link=logit))
res.logit.2
summary(res.logit.2)

# modell 3: mit 1 interaktion
res.logit.3 <- glm(factor.admit ~ gre + gpa*rank, 
                   family = binomial(link=logit))
res.logit.3
summary(res.logit.3)

# modell 4: mit 1 interaktion
res.logit.4 <- glm(factor.admit ~ gre + gpa + rank + gre:rank, 
                   family = binomial(link=logit))
res.logit.4
summary(res.logit.4)

# ROC-kurve
# (beispielhaft fuer modell 4)

# methode 1:
par(mfrow=c(1,1))
prob.pred = predict(res.logit.4,type=c("response"))
grad.data$prob.pred=prob.pred
roc.1 <- roc(factor.admit ~ prob.pred, data=grad.data)
plot(roc.1, col="lightslateblue") 

# methode 2:
modelfit <- glm(formula=factor.admit ~ gre + gpa + rank + gre:rank, 
                family=binomial(), data=grad.data, 
                na.action=na.omit)
rocplot(modelfit)

