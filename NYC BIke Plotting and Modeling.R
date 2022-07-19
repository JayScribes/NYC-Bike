---
  title: "NYC Bike Modeling"
author: "Jay"
date: '2022-07-18'
output: html_document
---
  ## Libraries
  ```{r}
library(tidyverse)
library(tidymodels)
library(skimr)
library(GGally)
library(geosphere)
library(doParallel)
library(themis)
library(DescTools)
library(baguette)
options(scipen=999)
```
## Loading the Data
```{r}
NYC_Bike <- read_csv("C:/Users/Dell/Desktop/Data Projects/Portfolio/Walkthroughs/NYC Bike Share/NYC_Bike.csv", 
                     col_types = cols(tripduration = col_number(), 
                                      ss_lat = col_number(), ss_lont = col_number(), 
                                      es_lat = col_number(), es_long = col_number(), 
                                      bikeid = col_character(), age = col_number(), 
                                      hour_of_day = col_number(), day_of_month = col_number(), 
                                      day_of_week = col_number(), month = col_number()))
```

## EDA & Cleaning
```{r}
skim(NYC_Bike)

NYC_Bike$outlier = NYC_Bike$age > 100    
NYC_Bike = filter(NYC_Bike, outlier != TRUE)
NYC_Bike <- subset(NYC_Bike, , -c(outlier))

NYC_Bike$age <-  Winsorize(NYC_Bike$age, minval = NULL, maxval = NULL, probs = c(0.04,0.96), na.rm = FALSE, type =1)

NYC_Bike$tripduration <-  Winsorize(NYC_Bike$tripduration, minval = NULL, maxval = NULL, probs = c(0.04,0.96), na.rm = FALSE, type =1)

```

## Data Validation
```{r}
NYC_Bike %>% 
  ggplot(aes(x=day_of_week))+
  geom_histogram()

NYC_Bike %>% 
  ggplot(aes(x=day_of_month))+
  geom_histogram()

NYC_Bike %>% 
  ggplot(aes(x=hour_of_day))+
  geom_histogram()

NYC_Bike %>% 
  ggplot(aes(x=month))+
  geom_histogram()
```

## Calculated Fields
```{r}
NYC_Bike =
  NYC_Bike %>% 
  rowwise() %>% 
  mutate(Distance = distHaversine(c(ss_lont, ss_lat), c(es_long, es_lat)))

NYC_Bike$Distance <-  Winsorize(NYC_Bike$Distance, minval = NULL, maxval = NULL, probs = c(0.04,0.96), na.rm = FALSE, type =1)

NYC_Bike$speed <- ((NYC_Bike$Distance/1000) / (NYC_Bike$tripduration/3600))


```

## Pairs Plots - Numerical Data
```{r}
NYC_Bike %>% 
  select(usertype, tripduration, age, hour_of_day, month) %>% 
  ggpairs(colums = 2:4, aes(color = usertype, alpha = 0.5))

NYC_Bike %>% 
  select(usertype, day_of_week, day_of_month, Distance, speed) %>% 
  ggpairs(colums = 2:4, aes(color = usertype, alpha = 0.5))
```

## Pairs Plots - Categorical Data
```{r}
NYC_Bike %>% 
  select(usertype, bikeid, gender, start_station_name, end_station_name) %>% 
  pivot_longer(bikeid:end_station_name) %>% 
  ggplot(aes(y = value, fill = usertype))+
  geom_bar(position = "fill")+
  facet_wrap(vars(name), scale = "free")+
  labs(x = NULL, y= NULL, fil = NULL)
```

## Transforming Day of Week to Labels
```{r}
NYC_Bike$day_of_week[NYC_Bike$day_of_week==1] <- "Sunday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==2] <- "Monday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==3] <- "Tuesday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==4] <- "Wednesday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==5] <- "Thursday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==6] <- "Friday"
NYC_Bike$day_of_week[NYC_Bike$day_of_week==7] <- "Saturday"

NYC_Bike$day_of_week <- as.character(NYC_Bike$day_of_week)
```

## Further Cleaning
```{r}
NYC_Bike$outlier = NYC_Bike$gender == "unknown"    
NYC_Bike = filter(NYC_Bike, outlier != TRUE)
NYC_Bike <- subset(NYC_Bike, , -c(outlier))
```


## Making Data Set For Classification
```{r}
data <- NYC_Bike %>% 
  select(tripduration, bikeid, age, gender, hour_of_day, day_of_month, day_of_week, month, Distance, speed, usertype)
```

## Splitting Data
```{r}
data_split <- initial_split(data, strata = usertype)
data_train <- training(data_split)
data_test <- testing(data_split)
```

## Resampling Folds
```{r}
bike_folds <- vfold_cv(data_train, v = 5, strata = usertype)
bike_folds

bike_metrics <- metric_set(accuracy, sensitivity, specificity, recall)
```

## Recipe
```{r}
bike_recipe <- recipe(usertype ~ ., data = data_train) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), - all_outcomes()) %>% 
  step_zv(all_predictors())

bike_recipe
```

## Generating Model
```{r}
bag_spec <-
  bag_tree(min_n = 10) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")
```

## Fitting Model Onto Data

```{r}
imb_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(bag_spec)

var.imp.t <- fit(imb_wf, data = data_train)
```

## Accounting For Imbalanced
```{r}
doParallel::registerDoParallel()
set.seed(123)
imb_results <- fit_resamples(
  imb_wf,
  resamples = bike_folds,
  metrics = bike_metrics
)

collect_metrics(imb_results)
```


## Upsampling
```{r}
bal_rec <- bike_recipe %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(usertype)


bal_wf <- workflow() %>% 
  add_recipe(bal_rec) %>% 
  add_model(bag_spec)

set.seed(123)
bal_results <- fit_resamples(
  bal_wf,
  resamples = bike_folds,
  metrics = bike_metrics,
  control = control_resamples(save_pred = TRUE))


collect_metrics(bal_results)

bal_results %>% 
  conf_mat_resampled()
```

## Fitting Onto Test Data
```{r}
bike_final <- bal_wf %>% 
  last_fit(data_split)

collect_metrics(bike_final)

collect_predictions(bike_final) %>% 
  conf_mat(usertype, .pred_class)
```

## Variable Importance Table
```{r}
var.imp.t
```


## ROC Curve

```{r}
bike_final %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(usertype, .pred_Customer) %>% 
  ggplot(aes(1 - specificity, sensitivity, color = id))+
  geom_abline(lty = 2, color = "gray90", size = 1.5)+
  geom_path(show.legend = FALSE, alpha = 0.6, size =1.2)+
  coord_equal()+theme_classic()
```

## Heatmap
```{r}
collect_predictions(bike_final) %>% 
  conf_mat(usertype, .pred_class) %>% 
  autoplot(cm, type = "heatmap")
```

