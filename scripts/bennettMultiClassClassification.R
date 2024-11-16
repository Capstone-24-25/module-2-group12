## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(e1071)

# load cleaned data
load('data/claims-clean-example.RData')

# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(mclass)

# If having library conflicts
#install.packages("keras", type = "source")
#library(keras)
#install_keras()

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

# Fit the preprocessing layer to the training text
preprocess_layer %>% adapt(train_text)

# Transform text to TF-IDF features
train_tfidf <- preprocess_layer(train_text) %>%
  as.matrix()

# Train SVM Model
svm_model <- svm(
  x = train_tfidf,
  y = as.factor(train_labels),
  kernel = "linear", # Use "radial" for a non-linear model
  type = "C-classification" # Multi-class SVM
)

# Save the trained model
saveRDS(svm_model, file = "results/bennett-multiclass-model.rds")