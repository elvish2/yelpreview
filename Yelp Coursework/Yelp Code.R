# Load the required libraries
library(glmnet)
library(rpart)
library(Metrics)
library(dplyr)
library(ggplot2)
library(tm)
library(sentimentr)
library(MLmetrics)
library(reshape2)
library(rpart.plot)



# Setup Working directories XXX/Small Dataset
setwd("C:/Users/Elvis/Desktop/Yelp Coursework")

load("yelp_review_small.Rda")
load("yelp_user_small.Rda")



################## Data Cleaning ###################

# Creating user ratios
user_data_small$useful_count_ratio <- round(as.numeric(user_data_small$useful/user_data_small$review_count),5)
user_data_small$funny_count_ratio <- round(as.numeric(user_data_small$funny/user_data_small$review_count), 5)
user_data_small$cool_count_ratio <- round(as.numeric(user_data_small$cool/user_data_small$review_count),5)

# Convert date into days of experience
user_data_small$yelping_since <- as.Date(user_data_small$yelping_since)

current_date <- Sys.Date()
user_data_small <- user_data_small|>
  mutate(days_of_experience = as.numeric(difftime(current_date, yelping_since, units = "days")))


# Join both datasets based on user_id column
master <- review_data_small |>
  inner_join(user_data_small, by = "user_id")

# Remove unnecessary columns
master <- subset(master, select = -c(useful.x, cool.x,funny.x,
                                     useful.y, cool.y,funny.y,
                                     name, review_id, business_id, user_id,
                                     review_count, date, yelping_since,
                                     friends, elite))
                                     
# Remove empty cells 
master <- na.omit(master)

# Checking data for missing observations
colSums(is.na(master))



#### Data Processing #########################################

# Remove outliers using Inter-Quartile Range.
# Create remove outlier function
outliers <- function(x) {
  
  Q1 <- quantile(x, probs=.25)
  Q3 <- quantile(x, probs=.75)
  iqr = Q3-Q1
  
  upper_limit = Q3 + (iqr*1.5)
  lower_limit = Q1 - (iqr*1.5)
  
  x > upper_limit | x < lower_limit
}

remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]),]
  }
  df
}

# Calling outlier function.
master_clean <- remove_outliers(master, c('useful_count_ratio', 'funny_count_ratio', 
                                       'cool_count_ratio', 'fans', 
                                       'compliment_hot',
                                       'compliment_more', 'compliment_profile', 
                                       'compliment_cute', 'compliment_list', 
                                       'compliment_note', 'compliment_plain',
                                       'compliment_cool', 'compliment_funny', 
                                       'compliment_writer', 'compliment_photos',
                                       'days_of_experience'))


# Sentiment Analysis (done after outlier to reduce computing load time)
master_clean$text <- tolower(master_clean$text)  # Convert text to lowercase
master_clean$text <- removePunctuation(master_clean$text)  # Remove punctuation
master_clean$text <- removeNumbers(master_clean$text)  # Remove numbers
corpus <- Corpus(VectorSource(master_clean$text)) # Create a corpus to remove stopwords
corpus <- tm_map(corpus, removeWords, stopwords("en"))# Remove stopwords
master_clean$text <- sapply(corpus, as.character)# Remove stopwords
master_clean$text <- stripWhitespace(master_clean$text)  # Remove extra whitespaces

sentiment_scores <- sentiment(master_clean$text)$sentiment # Calculate Sentiment Scores
master_clean$sentiment_score <- sentiment_scores # Adding score as new column
master_clean <- subset(master_clean, select = -c(text)) 


## Data Modelling #####################################

# Split the data into training and testing sets
set.seed(1)
train_index <- sample(1:nrow(master_clean), 0.9 * nrow(master_clean))  # 90% for training
train_data <- master_clean[train_index, ]
test_data <- master_clean[-train_index, ]

# Speciy predictor and target variable.
predictor_variables <- setdiff(names(train_data), "stars")

# Train Lasso Regression model
set.seed(1)
lasso_model <- cv.glmnet(as.matrix(train_data[, predictor_variables]), 
                         train_data[["stars"]], 
                         alpha = 1)

# Train Ridge Regression model
set.seed(1)
ridge_model <- cv.glmnet(as.matrix(train_data[, predictor_variables]), 
                         train_data[["stars"]], 
                         alpha = 0)

# Train Decision Tree Regression model
set.seed(1)
dt_model <- rpart(stars ~ ., data = train_data, method = "anova")

# Plot regression results
plot(lasso_model)
plot(ridge_model)
rpart.plot::rpart.plot(dt_model)



## Model performance analysis ########################################

# Make predictions on the test set
lasso_pred <- predict(lasso_model, newx = as.matrix(test_data[, predictor_variables]), s = "lambda.min")
ridge_pred <- predict(ridge_model, newx = as.matrix(test_data[, predictor_variables]), s = "lambda.min")
dt_pred <- predict(dt_model, newdata = test_data)

# Evaluate model performance
lasso_r2 <- R2_Score(lasso_pred, test_data$stars)
ridge_r2 <- R2_Score(ridge_pred, test_data$stars)
dt_r2 <- R2_Score(dt_pred, test_data$stars)

lasso_mae <- mae(lasso_pred, test_data$stars)
ridge_mae <- mae(ridge_pred, test_data$stars)
dt_mae <- mae(dt_pred, test_data$stars)

lasso_rmse <- rmse(lasso_pred, test_data$stars)
ridge_rmse <- rmse(ridge_pred, test_data$stars)
dt_rmse <- rmse(dt_pred, test_data$stars)

# Print the values obtained.
cat("Lasso Regression - R-squared:", lasso_r2, "MAE:", lasso_mae, "RMSE:", lasso_rmse, "\n")
cat("Ridge Regression - R-squared:", ridge_r2, "MAE:", ridge_mae, "RMSE:", ridge_rmse, "\n")
cat("Decision Tree Regression - R-squared:", dt_r2, "MAE:", dt_mae, "RMSE:", dt_rmse, "\n")

# Plot comparison
models <- c("Lasso", "Ridge", "Decision Tree")
r_squared <- c(lasso_r2, ridge_r2, dt_r2)
mae_values <- c(lasso_mae, ridge_mae, dt_mae)
rmse_values <- c(lasso_rmse, ridge_rmse, dt_rmse)

comparison_data <- data.frame(cbind(models, r_squared, mae_values, rmse_values))

colnames(comparison_data) <- c("Models", "R Square", "MAE", "RMSE")

comparison_data$'R Square' <- round(as.numeric(comparison_data$`R Square`), 3)
comparison_data$MAE <- round(as.numeric(comparison_data$MAE), 3)
comparison_data$RMSE <- round(as.numeric(comparison_data$RMSE), 3)

data_long <- melt(comparison_data, id.vars = "Models")

compare_model <- ggplot(data_long, aes(x = Models, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", width = 0.7) +
  labs(title = "Model Comparison",
       subtitle = "R-squared, MAE, and RMSE",
       y = "Metric Value") +
  theme_minimal() +
  theme(legend.position = "top")

print(compare_model)



##Save files#####################################################

## SAVING rds files for markdown
saveRDS(dt_model, file = "decision_tree_model.rds")
saveRDS(lasso_model, file = "lasso_model.rds")
saveRDS(ridge_model, file = "ridge_model.rds")
saveRDS(compare_model, file = "compare_model.rds")
