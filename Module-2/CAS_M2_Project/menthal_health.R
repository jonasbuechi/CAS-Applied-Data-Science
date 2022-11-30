# library("RSQLite")
# 
# ## connect to db
# con <- dbConnect(drv=RSQLite::SQLite(), dbname="~/Dropbox/Uni/CASADS22/git/CAS-Applied-Data-Science/Module-2/CAS_M2_Project/data/mental_health.sqlite")
# 
# ## list all tables
# tables <- dbListTables(con)

library(tidyverse)
Answer <- read_csv("Module-2/CAS_M2_Project/data/Answer.csv")
Question <- read_csv("Module-2/CAS_M2_Project/data/Question.csv")
Survey <- read_csv("Module-2/CAS_M2_Project/data/Survey.csv")

data_long <- full_join(Question, Answer, by = c("questionid" = "QuestionID"))
#select(-questionid)




data_wide <- data_long %>% 
  filter(!questionid %in% c(115, 116, 117)) %>% 
  pivot_wider(names_from = questiontext,
              values_from = c(AnswerText),
              id_cols = c(UserID, SurveyID)) %>% 
  janitor::clean_names()


data_wide_sub <- data_wide %>% 
  filter(what_country_do_you_work_in == "United States of America") %>% 
  mutate(age = as.numeric(what_is_your_age),
         year = factor(survey_id),
         sex = as_factor(what_is_your_gender) %>% 
           fct_collapse(Male = c("Male", "male"),
                        Female = c("Female", "female")) %>% 
           fct_lump_n(2),
         mental_health_disorder = as_factor(do_you_currently_have_a_mental_health_disorder) %>% 
           fct_collapse(Possibly = c("Don't Know", "Maybe", "Possibly")) %>% 
           fct_relevel(c("Yes", "Possibly"))) %>% 
  filter(!is.na(do_you_currently_have_a_mental_health_disorder)) %>% 
  filter(age > 18, age <= 67) # 67 is full retirement age in USA

data_wide_sub %>% 
  group_by(year) %>% 
  summarise(age_mean = mean(age),
            age_median = median(age),
            age_sd = sd(age),
            age_min = min(age),
            age_max = max(age),
            n = n())

data_wide_sub %>% 
  group_by(year, mental_health_disorder) %>% 
  summarise(n = n()) %>% 
  mutate(prop = n / sum(n)) %>% 
  ggplot(aes(fill = mental_health_disorder, x = year, y = prop))+
  geom_bar(position="fill", stat="identity")+
  geom_text(aes(label = scales::percent(round(prop, 2))), position = position_stack(vjust = 0.5))+
  #geom_line(aes(x = year, y = prop, group = mental_health_disorder), position = position_stack(vjust = 0.5))+
  #geom_point(aes(x = year, y = prop, group = mental_health_disorder), position = position_stack(vjust = 0.5))+
  ggtitle("Distribution of mental health disorder across questionnaires")+
  ylab("Proportion")+
  scale_y_continuous(
    labels = scales::percent_format()
  )+
  #viridis::theme_ipsum() +
  xlab("")+
  labs(fill = "Mental Health Disorder")+
  theme_minimal()+
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )


data_wide_sub %>% 
  group_by(year, mental_health_disorder) %>% 
  summarise(n = n()) %>% 
  mutate(prop = n / sum(n)) %>% 
  ggplot()+
  geom_line(aes(x = year, y = prop, group = mental_health_disorder))


data_wide_sub %>% 
  group_by(year, sex) %>% 
  summarise(n = n()) %>% 
  mutate(prop = n / sum(n)) %>% 
  ggplot(aes(fill = sex, x = year, y = prop))+
  geom_bar(position="fill", stat="identity")+
  geom_text(aes(label = scales::percent(round(prop, 2))), position = position_stack(vjust = 0.5))+
  #viridis::scale_fill_viridis(discrete = T, option = "D") +
  ggtitle("Distribution of sex across questionnaires") +
  ylab("Proportion")+
  scale_y_continuous(
    labels = scales::percent_format()
  )+
  #viridis::theme_ipsum() +
  xlab("")+
  labs(fill = "Sex")+
  theme_minimal()+
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )


data_wide_sub %>% 
  ggplot(aes(x = age))+
  ggtitle("") +
  ylab("Count")+
  xlab("Age")+
  geom_histogram()+
  facet_grid(~year)+
  theme_minimal()




not_all_na <- function(x) any(!is.na(x))
not_any_na <- function(x) all(!is.na(x))
data_wide %>%
  filter(survey_id == 2019) %>%
  select(where(not_all_na)) %>% 
  filter(!is.na(what_is_your_gender))


