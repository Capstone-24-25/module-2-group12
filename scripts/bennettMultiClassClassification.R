## PREPROCESSING
## Commented Out Because Already Completed
#source('scripts/preprocessing.R')
# load('data/claims-raw.RData')
#claims_clean <- claims_raw %>% parse_data()
#save(claims_clean, file = 'data/bennett-claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(e1071)
library(keras)

# load cleaned data
load("data/bennett-claims-clean-example.RData")

# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(mclass)

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split ="whitespace",
  ngrams = NULL,
  max_tokens = 5000, # Example token limit
  output_mode = "tf_idf"
)


# Fit the preprocessing layer to the training text
preprocess_layer %>% adapt(train_text)

# Preprocess text data
train_tfidf <- as.array(preprocess_layer(train_text))

# Train SVM Model
svm_model <- svm(
  x = train_tfidf,
  y = as.factor(train_labels),
  kernel = "linear",
  type = "C-classification"
)

# Save Model
saveRDS(svm_model, file = "results/bennett-multiclass-model.rds")
