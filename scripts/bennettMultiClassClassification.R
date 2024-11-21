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
library(dplyr)

# load cleaned data
load("data/claims-clean.RData")


# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(mclass)

# Test text for later
test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels <- testing(partitions) %>%
  pull(mclass)
test_ids <- testing(partitions) %>%
  pull(.id)

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
# saveRDS(svm_model, file = "results/bennett-multiclass-model.rds")

# Load test data
load("data/claims-test.RData")

# Preprocess test data using the preprocessing layer
test_tfidf <- as.array(preprocess_layer(test_text))

# Make predictions on the test data
test_predictions <- predict(svm_model, newdata = test_tfidf)

# Create the predictions tibble
pred_df <- tibble(
  .id = test_ids,
  mclass.pred = test_predictions,
  mclass = test_labels
)

# Save predictions to an RData file
save(pred_df, file = "results/preds-group12.RData")


## For Checking Accuracy
# Add a column for correctness
pred_df <- pred_df %>%
 mutate(correct = test_predictions == test_labels)

# Save updated predictions to a CSV file
write_csv(pred_df, "results/bennett-multiclass-predictions-with-accuracy.csv")

## Print overall accuracy
# Compare predictions with actual test labels
correct_predictions <- test_predictions == test_labels

# Calculate accuracy (convert logical to numeric explicitly)
accuracy <- mean(as.numeric(correct_predictions))

# Print the accuracy
print(glue::glue("Model Accuracy: {accuracy * 100}%"))

# Convert to factors if not already
multiclass_pred_df <- pred_df %>%
  mutate(
    mclass = as.factor(mclass),
    mclass.pred = as.factor(mclass.pred)
  )

# # Calculate accuracy
accuracy_result <- accuracy(multiclass_pred_df, truth = mclass, estimate = mclass.pred)

# # Calculate sensitivity for each class
sensitivity_result <- sens(multiclass_pred_df, truth = mclass, estimate = mclass.pred)

# # Calculate specificity for each class
specificity_result <- spec(multiclass_pred_df, truth = mclass, estimate = mclass.pred)

# # Combine results into a single table
metrics_table <- bind_rows(
  accuracy_result,
  sensitivity_result,
  specificity_result
)

# Save metrics_table as an RDS file
saveRDS(metrics_table, "results/multiclass_metrics_table.rds")
