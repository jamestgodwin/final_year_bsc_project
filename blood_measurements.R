normalsdata = read.table("normals.txt", header = T) 
normalsdata$carrier <- rep(0, nrow(normalsdata)) 
carrierdata = read.table("carriers.txt", header = T) 
carrierdata$carrier <- rep(1, nrow(carrierdata))

data = rbind(normalsdata, carrierdata)
data$carrier = as.factor(data$carrier)
#creating a new dataframe, merging the two datasets, and adding a new factor variable whether the individual is a carrier or not

datapca = prcomp(data[,1:6], scale = TRUE)
biplot(datapca)
plot(datapca$x[,1], datapca$x[,2], xlab = "PC1", ylab = "PC2", main = "PCA scores plot", pch =16, col = data$carrier)
legend("bottomright", inset = 0.01, legend = c("Normal", "Carrier"), col = unique(data$carrier),pch = 16)
#scaled PCA analysis of the dataset, however there is not much separation, maybe not best method?
#we also see there are probably no outliers

corrvariables = cor(data[,1:7], use = "all.obs", method = "pearson")
heatmap3(corrvariables, rowV = NULL, colv = NULL, showColDendro = F, showRowDendro = F,scale = "none", revC = T, key.title = "Correlation coefficient")
#heatmap shows most correlation is between carrier and m4, additonally m1 and m3 are quite correlated
#there is not much correlation between the date or any other variables #there is however some correlation between age and carrier

m = nrow(data)
trainIndex = sample(1:m, size = round(0.7*m), replace = FALSE) 
traindata = data[trainIndex ,]
testdata = data[-trainIndex ,]
#splitting the data into independent training and test data sets

summary(datapca)
#hence we should keep 5 PCs, since the cumulative variance is >0.9
pcascores = datapca$x[,1:5]
pcaldamodel = lda(pcascores, data[,7], CV = T)

table(real = data[,7], predicted = pcaldamodel$class)
#PCA-LDA model has an accuracy of ~0.874, good but not perfect, also will this generalise?

group = traindata[,7]
ldamodel = lda(traindata[,1:6], group)
ldamodel
#age has the greatest absolute value, and so is the most influential in distinguishing betweenthe groups
#m2 and m3 are also very good

ldamodelcv = lda(data[,1:6], data[,7], CV = T) 
table(predicted = ldamodelcv$class, real = data[,7]) 
#0.87 accuracy rate for CV lda model

predtest = predict(ldamodel, testdata[,1:6]) table(predicted = predtest$class, real = testdata[,7]) 
#~0.845 accuracy rating, less than the PCA-LDA model

predtrain = predict(ldamodel, traindata[,1:6])
table(predicted = predtrain$class, real = traindata[,7])
#~0.88 accuracy rating, hence the model is possibly overfitted

nb = train((carrier)~., method = "naive_bayes", data = data, trControl = trainControl(method = "LOOCV"))
nb$results

#we see that using a kernel is more accurate 
grid = expand.grid(laplace = 0, usekernel = F, adjust = 1)

nb = train((carrier)~., method = "naive_bayes", data = data, trControl = trainControl(method ="LOOCV"), preProcess=c("scale"), tuneGrid = grid) nb$results
#creation of a nb with required and wanted parameters

table(predicted = nb$pred[,1] , true = nb$pred[,2])
#we obtain a ~0.89 accuracy rating, our highest yet, but could be better

real = as.numeric(data[,7])
auc_nb = roc(real, as.numeric(nb$pred[,1])) 
auc_nb

nb$pred
newvarsnb = c("age", "m1", "m2", "m3", "m4", "carrier") 
newdatanb = data[newvarsnb]
newnb = train((carrier)~., method = "naive_bayes", data = newdatanb, trControl = trainControl(method = "LOOCV"), tuneGrid = grid)
pred_newnb = predict(newnb, newdatanb, type = "prob") 

rates_newnb = prediction(pred_newnb[,2], newdatanb[,6]) perf_newnb = performance(rates_newnb, "tpr", "fpr") 
plot(perf_newnb, col = "red", main = "ROC Curve for Naive Bayes")

model_nb = naive_bayes(traindata[,1:6], traindata[,7], usekernel = T) 
pred_nb = predict(model_nb, testdata[,1:6])
real = testdata[,7]
table(true = real, predicted = pred_nb)
#~0.879 accuracy for using test data

pred2_nb = predict(model_nb, traindata[,1:6])
real2 = traindata[,7]
table(true = real2, predicted = pred2_nb)
#using the training data we have an accuracy rating of ~0.92, hence poss overfit

real = as.numeric(testdata[,7])
auc_nb = roc(real, as.numeric(pred_nb))
auc_nb
#we find that the area under the ROC curve for the test data is 0.878

pred_nb = predict(model_nb, testdata[,1:6], type = "prob") 

library(ROCR)
pred_nb
rates_nb = prediction(pred_nb[,2], testdata[,7])
perf_nb = performance(rates_nb, "tpr", "fpr")
plot(perf_nb, col = "red", main = "ROC Curve for Naive Bayes") 
#plotting ROC curve for NB

plot(perf_nb, col = "blue", lty = 1, main = "ROC Curve for Naive Bayes, AUC = 0.8727") 
plot(perf_newnb, col = "red", add = TRUE, lty = 2)

pred_lda = predict(ldamodel, testdata[,1:6], type = "prob")
rates_lda = prediction(pred_lda$posterior[,2], testdata[,7])
perf_lda = performance(rates_lda, "tpr", "fpr")

plot(perf_lda, col = "blue", lty = 1, main = "ROC Curve Comparison of LDA and Naive Bayes") 
plot(perf_nb, col = "red", add = TRUE, lty = 2)
legend("bottomright", inset = 0.05, legend = c("LDA", "Naive Bayes"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 0.6)
#here we find that LDA is a better classifier for all sensistiivity settings and specificity settings

disease.tree = tree(carrier~., data = traindata) 
plot(disease.tree)
text(disease.tree, cex = 0.7)
#creation of decision tree

tree.pred = predict(disease.tree, traindata[,-7], type = "class") 
table(predicted = tree.pred, true = traindata[,7])
#training data has 0.948 accuracy

tree.pred = predict(disease.tree, testdata[,-7], type = "class") 
table(predicted = tree.pred, true = testdata[,7])
#test data has 0.81 accuracy, poss overfitted

cv.disease = cv.tree(disease.tree, FUN = prune.misclass) 
plot(cv.disease)
#hence we find we can prune the decision tree to 5

prune.disease = prune.misclass(disease.tree, best = 5) 
prune.disease
plot(prune.disease)
text(prune.disease, cex = 0.7)
#plotting new decision tree

prunetree.pred = predict(prune.disease, traindata[,-7], type = "class") 
table(predicted = prunetree.pred, true = traindata[,7])
#accuracy rate of 0.94 accuracy

prunetree.pred = predict(prune.disease, testdata[,-7], type = "class") 
table(predicted = prunetree.pred, true = testdata[,7])
#accuracy rate of 0.83, an improvement but not large

rf = randomForest(carrier~., data = traindata, mtry = 6, importance = TRUE) 
rf
#random forest model with OOB estimate of 9.56%

rf.pred = predict(rf, testdata[,-7], type = "class")
table(predicted = rf.pred, true = testdata[,7])
#again an accuracy rate of 0.83, and so the random forest classifier is not good

stepmodel = stepclass(carrier~., data = data, method = "lda", direction = "forward", criterion = "AC", improvement = 0.01)
#we can create a model that has m2, m3, m4, age #this gives an accuracy of 0.67
stepmodel

newvars = c("age", "m2", "m3", "m4")
newdata = data[newvars]
limmodel = lda(newdata, data[,7], CV = T)
table(predicted = limmodel$class, real = data[,7])
#0.88 accuracy rating in classification, hence using lda with LOO the accuracy is higher than using stepclass

pred_lda2 = limmodel$posterior[,2]
rates_lda2 = prediction(limmodel$posterior[,2], data[,7])
perf_lda2 = performance(rates_lda2, "tpr", "fpr")
plot(perf_lda2, col = "blue", lty = 1, main = "ROC Curve of LDA with stepwise") 
#roc curve for lda with stepwise variables

stepmodelback = stepclass(carrier~., data = data, method = "lda", direction = "backward", criterion = "AC", improvement = 0.01)
#backwards model

newvars2 = c("age", "m1", "m2", "m3", "m4")
newdata2 = data[newvars2]
limmodel2 = lda(newdata2, data[,7], CV = T)
table(predicted = limmodel2$class, real = data[,7])
#0.88 accuracy rating, more complex model so we prefer the forwards classification

qdamodel1 = qda(data[,-7], data[,7], CV = T) 
table(predicted = qdamodel1$class, real = data[,7]) 
#normal qda, all variables and a ~0.91 accuracy

rates_qda1 = prediction(qdamodel1$posterior[,2], data[,7]) 
pred_qda1 = qdamodel1$posterior[,2]
perf_qda1 = performance(rates_qda1, "tpr", "fpr")
plot(perf_qda2, col = "blue", lty = 1, main = "ROC Curve of QDA") 
#roc curve for QDA LOO CV

stepmodel2backward = stepclass(carrier~., data = data, method = "qda", direction = "forward", criterion = "AC", improvement = 0.01)
newvarsqda = c("age", "m1", "m2", "m3", "m4")
newdataqda = data[newvarsqda]
#best set of variables for using with qda, both directions produce the same combination

qdamodel = qda(newdataqda, data[,7], CV = T)
table(predicted = qdamodel$class, real = data[,7])
#0.91 accruacy rating, some improvement using a qda model and LOO cross validation

pred_qda2 = qdamodel$posterior[,2]
rates_qda2 = prediction(qdamodel$posterior[,2], data[,7])
perf_qda2 = performance(rates_qda2, "tpr", "fpr")

plot(perf_qda2, col = "blue", lty = 1, main = "ROC Curve of QDA with stepwise") #ROC curve for QDA with special subset of variables from stepwise
plot(perf_lda, col = "blue", lty = 1, main = "ROC Curve Comparison of QDA, LDA and Naive Bayes")
plot(perf_nb, col = "red", add = TRUE, lty = 2)
plot(perf_qda, col = "green", add = TRUE, lty = 3)
legend("bottomright", inset = 0.05, legend = c("LDA", "Naive Bayes", "QDA"), col = c("blue", "red", "green"), lty = 1:3, text.font = 2, cex = 0.6)
#comparison of ROC curve for QDA, LDA and Naive Bayes, and we see that LDA is best

plot(perf_qda1, col = "blue", lty = 1, main = "ROC Curve Comparison for QDA") plot(perf_qda2, col = "red", add = TRUE, lty = 2)
legend("bottomright", inset = 0.01, legend = c("All variables (AUC = 0.9325)", "Stepwise
variables (AUC = 0.9331)"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 1) 
#comparison of using qda for without/with stepwise collection of variables

realqda = as.numeric(data[,7])
auc_qda = roc(realqda, as.numeric(pred_qda1)) 
#all variables auc_qda

reallqa2 = as.numeric(data[,7])
auc_qda2 = roc(realqda, as.numeric(pred_qda2)) 
#new variables auc_qda2

plot(perf_lda, col = "blue", lty = 1, main = "ROC Curve Comparison for LDA") 
plot(perf_lda2, col = "red", add = TRUE, lty = 2)
legend("bottomright", inset = 0.01, legend = c("All variables (AUC = 0.9628)", "Stepwise variables (AUC = 0.9309)"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 1) #comparison of using lda with/without stepwise collection of variables
reallda = as.numeric(data[,7])
auc_lda = roc(reallda, as.numeric(pred_lda2))
auc_lda

reallda2 = as.numeric(testdata[,7])
auc_lda2 = roc(reallda2, as.numeric(pred_lda$posterior[,2])) 
auc_lda2
#calculation of area under the curve for LDA

datanewdate = data
datanewdate$date = sub('.*(\\d{2}).*', '\\1', datanewdate$date) #extracting last 2 digits of date ie the year
datanewdate = datanewdate[order(datanewdate$date),] rownames(datanewdate) <- 1:nrow(datanewdate)
#ordering dataset by new date and then changing row index

datanewdate$date = as.numeric(datanewdate$date) datanewdate$carrier = as.factor(datanewdate$carrier)

corrvariables2 = cor(datanewdate[,1:7], use = "all.obs", method = "pearson") 
heatmap3(corrvariables2, rowV = NULL, colv = NULL, showColDendro = F, showRowDendro = F, scale = "none", revC = T) 
#heat map of variables

datapca78 = prcomp(datanewdate[,1:6], scale = TRUE)
plot(datapca78$x[1:64,1], datapca78$x[1:64,2], xlab = "PC1", ylab = "PC2", main = "PCA scores plot - 1978", pch = 16, col = "red")
points(datapca78$x[1:39,1], datapca78$x[1:39, 2], pch = 16, col = "black") legend("bottomright", inset = 0.01, legend = c("Normal", "Carrier"), col = unique(datanewdate$carrier), pch = 16) 
#plot of 1978

plot(datapca78$x[65:194,1], datapca78$x[65:194,2], xlab = "PC1", ylab = "PC2", main = "PCA scores plot - 1979", pch = 16, col = "red")
points(datapca78$x[65:152,1], datapca78$x[65:152, 2], pch = 16, col = "black") legend("bottomright", inset = 0.01, legend = c("Normal", "Carrier"), col =unique(datanewdate$carrier), pch = 16) 
#plot for 1979 observations

plot(perf_qda2, col = "blue", lty = 1, main = "ROC Curve Comparison")
plot(perf_lda, col = "red", add = TRUE, lty = 2)
plot(perf_nb, col = "black", add = TRUE, lty = 3)
legend("bottomright", inset = 0.05, legend = c("QDA (AUC = 0.9331)", "LDA (AUC = 0.9628)", "Naïve Bayes (AUC = 0.8727)"), col = c("blue", "red", "black"), lty = 1:3, text.font = 2, cex= 0.6)
#roc curve comparison

costqda2 = performance(rates_qda2, 'ecost')

plot(costqda2, col = "blue", lty = 1, main = "Cost Curve Comparison") costlda = performance(rates_lda, 'ecost')
plot(costlda)

costnb = performance(rates_nb, 'ecost')
plot(costnb)
#cost curve analysis

plot(costqda2, col = "blue", lty = 1, main = "Cost Curve Comparison", xlab = "Probability Cost Function")
plot(costlda, col = "red", add = TRUE, lty = 2)
plot(costnb, col = "black", add = TRUE, lty = 3)
legend("bottom", inset = 0.05, legend = c("QDA", "LDA", "Naïve Bayes"), col = c("blue", "red",
"black"), lty = 1:3, text.font = 2, cex = 0.6) #cost curve comparison
