# Load final model
final_model <- readRDS('results/final-logistic.rds')

# Preprocess test data
clean_test <- claims_test %>%
  parse_data() %>%
  nlp_fn_bigrams()

# Apply PCA transformations
pca_test_unigrams <- predict(pca_unigrams, clean_test)
pca_test_bigrams <- predict(pca_bigrams, clean_test)

# Predict using final model
log_odds_test_unigrams <- predict(logistic_unigrams, newdata = pca_test_unigrams, type = 'response')
combined_test <- cbind(log_odds_test_unigrams, pca_test_bigrams)

pred_classes <- predict(final_model, newdata = combined_test, type = 'response') > 0.5

# Save predictions
pred_df <- claims_test %>%
  bind_cols(bclass.pred = pred_classes) %>%
  select(.id, bclass.pred)

