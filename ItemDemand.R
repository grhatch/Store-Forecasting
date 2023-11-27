library(tidyverse)
library(tidymodels)
library(vroom)
library(timetk)
library(patchwork)


train <- vroom("./STAT348/Store-Forecasting/demand-forecasting-kernels-only/train.csv") 
test <- vroom("./STAT348/Store-Forecasting/demand-forecasting-kernels-only/test.csv")
View(train)






#################
## Time Series ##
#################

# trend: overall season-over-season pattern
# season length: length of a single cycle
# seasonal variation: cycle within a single season
# stationary process: time series whose properties (trend, seasonal variation, autocorrelation, etc.) don't change over time
# forecast: predict forward in time


nStores <- max(train$store)
nItems <- max(train$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)

    ## Fit storeItem models here
    
    ## Predict storeItem sales
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

storeItemTrain %>%
  plot_time_series(train$date, train$sales, .interactive=FALSE)





plot1 <- train %>%
  filter(store == 1, item == 1) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365) + ggtitle('store 1, item 1')
  
plot2 <- train %>%
  filter(store == 2, item == 4) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365) + ggtitle('store 2, item 4')

plot3 <- train %>%
  filter(store == 3, item == 2) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365) + ggtitle('store 3, item 2')

plot4 <- train %>%
  filter(store == 4, item == 3) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365) + ggtitle('store 4, item 3')

grid_plot <- plot1 + plot2 + plot3 + plot4
grid_plot


#################################

#################################

train36 <- train %>%
  filter(store == 3, item == 6)

View(train36)

rf_recipe <- recipe(sales~., data=train36) %>%
  step_rm(store, item) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month") %>%
  step_date(date, features="year") %>%
  step_date(date, features="doy") %>%
  step_date(date, features="decimal") %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))
  
#step_lag(vble, lag=howLong)




prep <- prep(rf_recipe)
baked <- bake(prep, new_data = train36)


rf_model <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")
  

# set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

L <- 5
## Grid of values to tune over; these should be params in the model
rf_tuning_grid <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
rf_folds <- vfold_cv(train36, v = K, repeats=1)

## Run CV
rf_CV_results <- rf_wf %>%
  tune_grid(resamples=rf_folds,
            grid=rf_tuning_grid,
            metrics=metric_set(smape))


## Find Best Tuning Parameters
rf_bestTune <- rf_CV_results %>%
  select_best("smape")



# turn this into LS
mean <- collect_metrics(rf_CV_results) %>%
  filter(mtry == rf_bestTune$mtry, min_n == rf_bestTune$min_n, .config == rf_bestTune$.config)

mean


## Finalize the Workflow & fit it
rf_final_wf <-
  rf_wf %>%
  finalize_workflow(rf_bestTune) %>%
  fit(data=train)

## Predict
rf_pred <- rf_final_wf %>%
  predict(new_data = test, type="class") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_class) %>% # Just keep datetime and predictions
  rename(type = .pred_class) # rename pred to count (for submission to Kaggle)

vroom_write(rf_pred, "rf_itemDemand.csv", delim = ',')

  

########################
## Time Series Models ##
########################

# eponential smoothing
# (S)ARIMA Models

# order matters in TS Models
# can't randomly split for CV
      # we can only chop the end off


library(modeltime) #Extensions of tidymodels to time series
library(timetk) #Some nice time series functions

train36 <- train %>%
  filter(store == 1, item == 6)

cv_split <- time_series_split(train36, assess="3 months", cumulative = TRUE) #cut off 3 most recent months
combo2_cutoff <- cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE) #plot with cut off piece


es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

## Visualize CV results
combo2_results <- cv_results %>%
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train36) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)



combo1_cutoff
combo1_results
combo2_cutoff
combo2_results

plotly::subplot(combo1_cutoff, combo2_cutoff, combo1_results, combo2_results, nrows=2)


