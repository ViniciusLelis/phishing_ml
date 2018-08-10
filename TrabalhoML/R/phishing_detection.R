library(RSNNS)
library(doParallel)

# Registramos computação paralela
# O número de "workers" é igual a metade do número de núcleos do processador da máquina
registerDoParallel()

# Carregando os dados a partir do arquivo CSV phishing_websites contendo as informações sobre os sites
# Esse arquivo deve estar na pasta raiz do projeto dentro da pasta Dataset
data <- read.csv(file="Dataset/phishing_websites.csv", header=FALSE, sep=",")

# Nome das colunas, ou seja, o nome dos atributos dos dados.
# O último atributo (target) informa se o site é phishing ou não (1 para phishing e -1 para seguro)
atributos <- c("ip_como_url", "url_longa", "encurtador_links", "tem_arroba",
           "redireciona_barra_barra", "prefx_sufx_hifen", "sub_dominio",
           "estado_ssl", "duracao_dominio", "favicon", "porta_nao_padrao",
           "https_token", "url_requisicao", "cond_ancora", "tag_links",
           "SFH", "pede_email", "url_anormal", "redireciona",
           "status_bar", "click_direito", "popup", "iframe_redirec",
           "idade_dominio", "dns_registro", "trafego", "rank_pagina",
           "indexado_google", "links_para_pagina", "relatorios_estatisticos", "target")

# Adicionamos à tabela de dados o nome das colunas
names(data) <- atributos

# Embaralhamos os dados utilizados
data <- data[sample(1:nrow(data),length(1:nrow(data))),1:ncol(data)]

# Separamos os dados do target de cada um
dataValues <- data[,1:30]
dataTargets <- decodeClassLabels(data[,31])

# Aplicamos o algoritmo de K Fold Cross Validation com K = 10
k_cv <- 10
folds <- cut(seq(1, nrow(data)), breaks = k_cv, labels=FALSE)
accuracyList <- vector("numeric", k_cv)
confusionMatrixList <- vector("list", k_cv)
sensitivityList <- vector("numeric", k_cv)
specificityList <- vector("numeric", k_cv)

for (i in 1:k_cv) {
  cat("\nIniciando geração do modelo para k =", i, "\n")
  testIndexes <- which(folds==i, arr.ind = TRUE)
  testDataValues <- dataValues[testIndexes, ]
  testDataTargets <- dataTargets[testIndexes, ]
  trainDataValues <- dataValues[-testIndexes, ]
  trainDataTargets <- dataTargets[-testIndexes, ]

  model <- RSNNS::mlp(trainDataValues, trainDataTargets, size=c(12,12,12,12), learnFuncParams=c(0.1,0.2,0.1,0.2),
                      maxit=3000, learnFunc="Rprop", inputsTest=testDataValues, targetsTest=testDataTargets)

  predictions <- predict(model, testDataValues)

  myMatrix <- RSNNS::confusionMatrix(testDataTargets,predictions)
  accuracyPrediction <- (myMatrix[1, 1] + myMatrix[2, 2])/nrow(testDataTargets)
  sensitivity <- myMatrix[2, 2]/(myMatrix[2, 2] + myMatrix[2,1])
  specificity <- myMatrix[1, 1]/(myMatrix[1, 1] + myMatrix[1,2])

  confusionMatrixList[[i]] <- myMatrix
  accuracyList[i] <- accuracyPrediction
  sensitivityList[i] <- sensitivity
  specificityList[i] <- specificity

  cat("Matriz de confusao para k =", i, "\n")
  print(myMatrix)
  cat("A acurácia para k =", i, "é", accuracyPrediction, "\n")
  cat("A sensibilidade para k =", i, "é", sensitivity, "\n")
  cat("A especificidade para k =", i, "é", specificity, "\n")
}

averageAccuracy <- mean(accuracyList)
averageConfusionMatrix <- Reduce("+", confusionMatrixList) / length(confusionMatrixList)
averageSensitivity <- mean(sensitivityList)
averageSpecificity <- mean(specificityList)

cat("A acurácia final do modelo é", averageAccuracy, "\n")
cat("A matriz de confusão média do modelo é\n")
print(averageConfusionMatrix)
cat("A sensibilidade média do modelo é", averageSensitivity, "\n")
cat("A especificidade média do modelo é", averageSpecificity, "\n")
