# Source preprocessing functions
source('scripts/preprocessing.R')

# Load raw data
load('data/claims-raw.RData')

# Preprocess unigrams
claims_clean <- claims_raw %>%
  parse_data()

# Preprocess bigrams
claims_bigrams <- claims_clean %>%
  nlp_fn_bigrams()

# Save preprocessed data
save(claims_clean, file = 'data/claims-clean-unigrams.RData')
save(claims_bigrams, file = 'data/claims-clean-bigrams.RData')

# MODEL TRAINING (Logistic Regression with Unigrams)
####################################################
library(glmnet)

# Load unigram-cleaned data
load('data/claims-clean-unigrams.RData')

# Partition unigram data
set.seed(110122)
partitions_unigrams <- claims_clean %>%
  initial_split(prop = 0.8)

train_unigrams <- training(partitions_unigrams)

train_unigrams_numeric <- train_unigrams %>%
  select(where(is.numeric))

train_labels_unigrams <- training(partitions_unigrams) %>%
  pull(bclass) %>%
  as.numeric() - 1

# PCA on unigram features
pca_unigrams <- prcomp(train_unigrams_numeric, scale. = TRUE)
pc_train_unigrams <- as.data.frame(pca_unigrams$x)

# Fit logistic regression using principal components
logistic_unigrams <- glm(train_labels_unigrams ~ ., data = pc_train_unigrams, family = binomial)

# Save PCA model and logistic model
saveRDS(pca_unigrams, file = 'results/pca-unigrams.rds')
saveRDS(logistic_unigrams, file = 'results/logistic-unigrams.rds')

# Load bigram-cleaned data
load('data/claims-clean-bigrams.RData')

# Partition bigram data
partitions_bigrams <- initial_split(claims_bigrams, prop = 0.8)

# PCA on bigram features
train_bigrams <- training(partitions)

train_bigrams_numeric <- train_bigrams %>%
  select(where(is.numeric))

pca_bigrams <- prcomp(train_bigrams_numeric, scale. = TRUE)
pc_train_bigrams <- as.data.frame(pca_bigrams$x)

# Predict log-odds from unigram logistic regression
log_odds_unigrams <- predict(logistic_unigrams, newdata = pc_train_unigrams, type = 'response')
log_odds_unigrams <- data.frame(log_odds = log_odds_unigrams, row.names = rownames(pc_train_unigrams))

# Combine log-odds with bigram principal components
combined_train <- cbind(log_odds_unigrams, pc_train_bigrams)

# Save PCA model for bigrams
saveRDS(pca_bigrams, file = 'results/pca-bigrams.rds')

# MODEL TRAINING (Logistic Regression with Bigrams and Log-Odds)
#################################################################

# Combine log-odds from unigrams and bigram PCs
final_model <- glm(train_labels_unigrams ~ ., data = combined_train, family = binomial)

# Save final model
saveRDS(final_model, file = 'results/final-logistic.rds')

# Evaluate unigram model
aic_unigrams <- AIC(logistic_unigrams)

# Evaluate final model
aic_final <- AIC(final_model)

cat("AIC (Unigrams):", aic_unigrams, "\n")
cat("AIC (Final Model with Bigrams):", aic_final, "\n")
