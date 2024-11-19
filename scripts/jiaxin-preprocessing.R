
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
library(tidytext)
require(keras)
require(tensorflow)
load('data/claims-raw.RData')

library(glmnet)
library(rsample)
library(sparsesvd)
library(yardstick)

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
# load functions needed for this assignment
source(paste(url, 'projection-functions.R', sep = ''))


####### NO HEADERS #########
## function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>% #paragraph elements
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>% # remove url
    rm_email() %>% # remove email
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>% # lowercased
    str_replace_all("\\s+", " ")
}

parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}


nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token,
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

## preprocessing
cleaned_claims <- claims_raw %>% parse_data() 
rl_claims <- nlp_fn(cleaned_claims)

set.seed(203842)
## splitting the data
partition <- rl_claims %>% initial_split(prop = 0.8)

## separate DTM from labels
train_dtm <- training(partition) %>% select(-.id, -bclass, -mclass)
train_label <- training(partition) %>% select(.id, bclass, -mclass)

# test set
test_dtm <- testing(partition) %>% select(-.id, -bclass, -mclass)
test_label <- testing(partition) %>% select(.id, bclass, -mclass)

# find projections based on training data
project <- projection_fn(.dtm = train_dtm, .prop = 0.7) # reducing training DTM’s size
projected_train_dtm <- project$data # projects data onto principal components

project$n_pc # number of components were used

train <- train_label %>% 
  transmute(bclass = factor(bclass)) %>%
  bind_cols(projected_train_dtm)

# fitting a binary logistic regression model using principal components as predictors
fit <- glm(bclass~., data = train, family = binomial)

# prediction on test data:

# projecting test DTM onto training PCA space
projected_test_dtm <- reproject_fn(.dtm = test_dtm, project)
# compute probabilities
prediction <- predict(fit,newdata = as.data.frame(projected_test_dtm),
                      type = 'response')

# converting probabilities to binary labels
prediction_df <- test_label %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(prediction)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# model performance
class_metrics <- metric_set(yardstick::sensitivity, 
                            yardstick::specificity, 
                            yardstick::accuracy, 
                            yardstick::roc_auc)
prediction_df %>% class_metrics(trut = bclass, 
                                estimate = bclass.pred, 
                                pred, 
                                event_level = 'second')



####### WITH HEADERS #########
# function to parse html and clean text
pca_fn_headers <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>% # include headers
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>% 
    rm_email() %>% 
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>% 
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
pca_data_headers <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = pca_fn_headers(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(pca_data_headers.out){
  out <- pca_data_headers.out %>% 
    unnest_tokens(output = token,
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    # transforms data into wide version
    pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

cleaned_claims_headers <- claims_raw %>%
  pca_data_headers()
headers_clean <- cleaned_claims_headers %>%
  select(-c(1:5), -7)


headers_clean <- headers_clean %>% 
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words',
                stopwords = str_remove_all(stop_words$word, 
                                           '[[:punct:]]')) %>%
  mutate(token.lem = lemmatize_words(token)) %>%
  filter(str_length(token.lem) > 2) %>%
  count(.id, bclass, mclass, token.lem, name = 'n') %>%
  bind_tf_idf(term = token.lem, 
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)

set.seed(203842)
# splitting data
pca_header <- headers_clean %>% initial_split(prop = 0.8)

# separate DTM from labels
train_dtm_pca <- training(pca_header) %>% select(-.id, -bclass, -mclass)
train_label_pca <- training(pca_header) %>% select(.id, bclass, -mclass)

test_dtm_pca <- testing(pca_header) %>% select(-.id, -bclass, -mclass)
test_label_pca <- testing(pca_header) %>% select(.id, bclass, -mclass)

# get principle components
proj_out <- projection_fn(.dtm = train_dtm_pca, .prop = 0.8) 
projected_header_train_dtm <- proj_out$data # projects data onto principal components
proj_out$n_pc 

# Logistic Regression
train_header <- train_label_pca %>% 
  transmute(bclass = factor(bclass)) %>%
  bind_cols(projected_header_train_dtm)

# fitting a binary logistic regression model 
fit_header <- glm(bclass~., data = train_header, family = binomial)

# On Test Data
projected_test_dtm <- reproject_fn(.dtm = test_dtm_pca, proj_out)

# predicting probabilities
prediction_header <- predict(fit_header, newdata = as.data.frame(projected_test_dtm),
                             type = 'response')

# store predictions in binary class labels based on a threshold of 0.5.
prediction_df_header <- test_label_pca %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(prediction_header)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# define classification metric panel 
class_metrics <- metric_set(
                            yardstick::sensitivity, 
                            yardstick::specificity, 
                            yardstick::accuracy, 
                            yardstick::roc_auc)
# compute test set accuracy
prediction_df_header %>% class_metrics(truth = bclass, 
                                       estimate = bclass.pred, 
                                       pred, 
                                       event_level = 'second')
