newphagepca = prcomp(newphagedata[,2:24], scale = TRUE) 
summary(newphagepca)
#since the variables are collinear we need to produce some PCs to perform PCA-LDA 
#we find the first 11 PCs have 95% of the cumulative variance

ldamodel = lda(newphagepca$x[,1:11], newphagedata[,1], CV = T) 
#PCA-LDA model
table(predicted = ldamodel$class, real = newphagedata[,1]) #accuracy rating of 0.905

pred_lda = ldamodel$posterior[,2]
rates_lda = prediction(pred_lda, newphagedata[,1])
perf_lda = performance(rates_lda, "tpr", "fpr")
plot(perf_lda, col = "blue", lty = 1, main = "ROC Curve of LDA") 
#ROC curve for LDA

reallda = as.numeric(newphagedata[,1]) 
auc_lda = roc(reallda, as.numeric(pred_lda)) 
auc_lda
#AUC for LDA is 0.9706

stepmodel = stepclass(output~., data = newphagedata, method = "lda", direction = "forward", criterion = "AC", improvement = 0.001)
#stepwise variable selection

newvars = c("numCGs", "numCCs", "numACs", "propcg", "numGs", "numCs", "numTAs") 
newdata = newphagedata[newvars]
#new dataset for stepwise variables

ldamodel2 = lda(newdata, newphagedata[,1], CV = T) 
table(predicted = ldamodel2$class, real = newphagedata[,1]) 
#0.94 accuracy using the stepwise collection of variables

pred_lda2 = ldamodel2$posterior[,2]
rates_lda2 = prediction(pred_lda2, newphagedata[,1])
perf_lda2 = performance(rates_lda2, "tpr", "fpr")
plot(perf_lda2, col = "red", lty = 1, main = "ROC Curve for LDA with stepwise") 
#ROC Curve for LDA with stepwise

reallda2 = as.numeric(newphagedata[,1]) 
auc_lda2 = roc(reallda2, as.numeric(pred_lda2)) 
auc_lda2
#AUC for LDA with stepwise is 0.9823

plot(perf_lda, col = "blue", lty = 1, main = "ROC Curve Comparison for LDA")
plot(perf_lda2, col = "red", add = TRUE, lty = 2)
legend("bottomright", inset = 0.05, legend = c("PCA-LDA (AUC = 0.9706)", "LDA with stepwise (AUC = 0.9823)"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 1.3) 
#ROC curve comparison for both LDA methods

# ///////// ////////// ////////// QDA ////////// /////////// //////////
qdamodel = qda(newphagepca$x[,1:11], newphagedata[,1], CV = T) 
#qda model

table(predicted = qdamodel$class, real = newphagedata[,1]) 
#accuracy rating of 0.92

pred_qda = qdamodel$posterior[,2]
rates_qda = prediction(pred_qda, newphagedata[,1])
perf_qda = performance(rates_qda, "tpr", "fpr")
plot(perf_qda, col = "blue", lty = 1, main = "ROC Curve of QDA") #ROC curve for QDA

realqda = as.numeric(newphagedata[,1]) 
auc_qda = roc(realqda, as.numeric(pred_qda)) 
auc_qda
#AUC for QDA is 0.9754

stepmodelqda = stepclass(output~., data = newphagedata, method = "qda", direction = "forward", criterion = "AC", improvement = 0.001)
#stepwise variable selection

newvars2 = c("numCGs", "propcg", "numGs", "numCs", "numTTs", "numGCs", "numAGs","numGAs", "numACs", "numAs", "numCAs") 
newdata2 = newphagedata[newvars2]
#new dataset

qdamodel2 = qda(newdata2, newphagedata[,1], CV = T) 
table(predicted = qdamodel2$class, real = newphagedata[,1]) 
#accuracy 0.94

pred_qda2 = qdamodel2$posterior[,2]
rates_qda2 = prediction(pred_qda2, newphagedata[,1])
perf_qda2 = performance(rates_qda2, "tpr", "fpr")

plot(perf_qda2, col = "blue", lty = 1, main = "ROC Curve for QDA with stepwise") 
#ROC curve for QDA with stepwise

realqda2 = as.numeric(newphagedata[,1]) 
auc_qda2 = roc(realqda2, as.numeric(pred_qda2)) 
auc_qda2
#AUC for QDA with stepwise is 0.9791

plot(perf_qda, col = "blue", lty = 1, main = "ROC Curve Comparison for QDA") 
plot(perf_qda2, col = "red", add = TRUE, lty = 2)
legend("bottomright", inset = 0.05, legend = c("PCA-QDA (AUC = 0.9754)", "Stepwise QDA(AUC = 0.9791)"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 1.2) 
#ROC curve comparison for both QDA methods

costlda2 = performance(rates_lda2, 'ecost')
costqda2 = performance(rates_qda2, 'ecost')
plot(costlda2, col = "blue", lty = 1, main = "Cost Curve Comparison", xlab = "Probability Cost Function")
plot(costqda2, col = "red", add = TRUE, lty = 2)
legend("bottom", inset = 0.05, legend = c("LDA with Stepwise", "QDA with stepwise"), col = c("blue", "red"), lty = 1:2, text.font = 2, cex = 1) 
#cost curve comparison for optimal LDA and QDA methods

# /////// ////////// ////////// ENSEMBLE CLASSIFIERS ///////// ///////// ///////////
dna.tree = tree(output~., data = traindata) 
plot(dna.tree)
text(dna.tree, cex = 0.75)
dna.tree
#creation of deciison tree

tree.pred = predict(dna.tree, traindata[,-1], type = "class") 
table(predicted = tree.pred, true = traindata[,1])
#0.95 accuracy rating with training data

tree.pred = predict(dna.tree, testdata[,-1], type = "class") 
table(predicted = tree.pred, true = testdata[,1])
#0.938 accuracy rating, possibly overfit so should prune

cv.dna = cv.tree(dna.tree, FUN = prune.misclass) 
plot(cv.dna, cex = 1.2)
#the optimum number of branches is 4

prune.dna = prune.misclass(dna.tree, best = 2) 
plot(prune.dna)
text(prune.dna, cex = 1.2)
#plot of new pruned decision tree

tree.pred = predict(prune.dna, traindata[,-1], type = "class") 
table(predicted = tree.pred, true = traindata[,1])
#0.923 accuracy rating

tree.pred = predict(prune.dna, testdata[,-1], type = "class") 
table(predicted = tree.pred, true = testdata[,1])
 #0.911 accruacy, much less overfitted so is improved
 
 
rf = randomForest(output~., data = traindata, ntree = 200, mtry = 10, importance = TRUE) 
rf
#random forest model, we get accuracy of 0.93
#OOB error for training set is 7.38%

rf.pred = predict(rf, testdata[,-1], type = "class")
table(predicted = rf.pred, true = testdata[,1])
#accuracy of 0.96 using the test data
#error for test set is ~6.66%, therefore the model is not overfitting

adaboost = boosting(output~., data = traindata, control = rpart.control(maxdepth = 1)) 
predboosting = predict.boosting(adaboost, newdata = testdata[,-1])
table(predicted = predboosting$class, true = testdata[,1])
#0.938 accuracy for test data

predboosting2 = predict.boosting(adaboost, newdata = traindata[,-1]) 
table(predicted = predboosting2$class, true = traindata[,1])
#0.947 accuracy rating for training data
# /////// ///// DISCARDED METHODS //////// / /////////

# /////// ///// PLSR ///////// ///////// ///////
newphagedata.df = data.frame(X = I(as.matrix(newphagedata[,2:24])), Y = I(as.matrix(newphagedata[,1]))) 
#dataframe for PLS
plsrmodel = plsr(Y~X, data = newphagedata.df, ncomp = 20, scale = T, validation = "LOO", method = "kernelpls")
#PLSR model

plot(RMSEP(plsrmodel), legendpos = "topright")
selectNcomp(plsrmodel, plot = T)
#we find the min number of components is 10

plsrmodel = plsr(Y~X, data = newphagedata.df, ncomp = 10, scale = T, validation = "LOO", method = "kernelpls")
#PLSR model with optimal no. of components biplot(plsrmodel, comps = 1:2, which = "scores", cex = 0.6) 
#biplot of X scores and Y scores

biplot(plsrmodel, comps = 1:2, which = "y", cex = 0.6) 
#biplot of Y scores and Y loadings

plot(plsrmodel$scores[,1], plsrmodel$scores[,2], xlab = "PLS1", ylab = "PLS2", pch = 20, col=as.factor(newphagedata[,1]))

n = nrow(newphagedata.df)
trainIndex2 = sample(1:n, size = round(0.7*n), replace = FALSE) 
plstraindata = newphagedata.df[trainIndex2, ]
plstestdata = newphagedata.df[-trainIndex2,]
#training and test data sets using the df

plstrain = plsr(Y~X, data = plstraindata, ncomp = 10, scale = T, validation = "LOO", method = "kernelpls")
#pls model for training data
pred = predict(plstrain, ncomp = 10, newdata = plstestdata) predclass = as.integer(pred + 0.5)
table(predicted = predclass, true = plstestdata$Y)
#for the test data we have accruacy of 92%

pred = predict(plstrain, ncomp = 10, newdata = plstraindata)
predclass = as.integer(pred + 0.5)

table(predicted = predclass, true = plstraindata$Y)
#for the training data we have accuracy of 93%, poss overfit but not drastic

# //////// ////////// NEURAL NETWORKS ///////// ////////
nnmodel = nnet(output~., data = traindata, size = 6) nnpred = predict(nnmodel, testdata[,-1], type = "class") 
table(predicted = as.factor(nnpred), true = testdata[,1]) 
#standard neural network model
test.acc=double(10) for(i in 1:10)
{
nnmodel=nnet(output~., data=traindata, size=6)
nnpred= predict(nnmodel, testdata[,-1], type = "class") 
cm=table(predicted = as.factor(nnpred), true = testdata[,1]) 
test.acc[i]=(cm[1,1]+cm[2,2])/dim(testdata)[1]
}
#loop for 10 neural network models test.acc
#accuracy for each NN model sd(test.acc)
#sd for the 10 models

numFolds = trainControl(method = 'cv', number = 10, savePredictions = T)
nnmodel2 = train(output~., data = traindata, method = 'nnet', preProcess = c('center', 'scale'),
trControl = numFolds) nnmodel2
grid = expand.grid(size = 5, decay = 0.1)
nnmodel3 = train(output~., data = traindata, method = 'nnet', maxit = 100, preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid = grid) nnmodel3$results
#NN model with cross validation

table(predicted = nnmodel3$pred[,1], real = nnmodel3$pred[,2]) 
#accuracy of 0.945 with train data
nnpred = predict(nnmodel3, testdata[,-1]) 
table(predicted = nnpred, real = testdata[,1]) 
#0.92 with test data, clearly overfitted
