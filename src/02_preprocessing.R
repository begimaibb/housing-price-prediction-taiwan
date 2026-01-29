# ===============================
# 02_preprocessing.R
# Data Cleaning & Feature Engineering
# ===============================

library(dplyr)
library(readr)
library(lubridate)
library(stringr)

housing <- read_csv("data/RAW_new_taipei_housing_transactions_2018_2023.csv")

# -------------------------------
# Missing value handling
# -------------------------------

housing <- housing %>% select(-rps01, -rps12)

# Fill missing urban zoning values
missing_rows_rps04 <- which(is.na(housing$rps04))
housing$rps04[missing_rows_rps04[housing$rps05[missing_rows_rps04] != "NA"]] <- "non-urban"
housing$rps04[is.na(housing$rps04)] <- names(sort(table(housing$rps04), decreasing = TRUE))[1]

housing <- housing %>% select(-rps05, -rps06)

# Convert Taiwanese dates to standard dates
taiwanese_to_date <- function(x) {
  year <- x %/% 10000 + 1911
  month <- (x %% 10000) %/% 100
  day <- x %% 100
  ymd(paste(year, month, day, sep = "-"))
}

housing <- housing %>%
  mutate(
    rps14 = taiwanese_to_date(rps14),
    rps07 = taiwanese_to_date(rps07)
  ) %>%
  filter(!is.na(rps14), !is.na(rps10))

# Fill missing parking info
housing$rps23[is.na(housing$rps23)] <- "no parking"

# -------------------------------
# Outlier removal (IQR method)
# -------------------------------

cols <- c("rps03","rps15","rps21","rps22","rps24","rps25","rps30")

for (col in cols) {
  q1 <- quantile(housing[[col]], 0.25, na.rm = TRUE)
  q3 <- quantile(housing[[col]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  housing <- housing %>%
    filter(
      is.na(.data[[col]]) |
        (.data[[col]] >= lower_bound & .data[[col]] <= upper_bound)
    )
}


# -------------------------------
# Feature engineering
# -------------------------------

housing$age <- round(as.numeric(difftime(housing$rps07, housing$rps14, units = "days")) / 365.25)


housing <- housing %>%
  rename(
    district = district,
    address = rps02,
    landArea = rps03,
    landUseZoning = rps04,
    transactionDate = rps07,
    package = rps08,
    apartmentFloor = rps09,
    totalFloors = rps10,
    buildingType = rps11,
    buildingMaterials = rps13,
    constructionDate = rps14,
    buildingArea = rps15,
    room = rps16,
    livingroom = rps17,
    bathroom = rps18,
    divisionWall = rps19,
    management = rps20,
    totalPrice = rps21,
    sqmPrice = rps22,
    parkingCategory = rps23,
    parkingArea = rps24,
    parkingPrice = rps25,
    remarks = rps26,
    transactionNumber = rps27,
    mainBuildingArea = rps28,
    annexBuildingArea = rps29,
    balconyArea = rps30,
    elevator = rps31
  )


# -------------------------------
# Convert Chinese floor notation to numeric
# Single numeric floor is kept; multiple floors, whole-building (全) entries → NA
# -------------------------------

chinese_to_int <- function(x) {
  map <- c(
    "零" = 0, "一" = 1, "二" = 2, "三" = 3, "四" = 4,
    "五" = 5, "六" = 6, "七" = 7, "八" = 8, "九" = 9,
    "十" = 10
  )
  
  sapply(x, function(val) {
    if (is.na(val)) return(NA_real_)
    
    # Split by comma (， or ,)
    floors <- str_split(val, "[,，]")[[1]]
    
    # Remove descriptors like 層, 樓 (floor, building)
    floors <- str_remove_all(floors, "[層樓]")
    
    numeric_floors <- sapply(floors, function(f) {
      f <- str_trim(f)
      if (f == "十") return(10)
      if (str_detect(f, "十")) {
        parts <- str_split(f, "十")[[1]]
        tens <- ifelse(parts[1] == "", 1, map[parts[1]])
        ones <- ifelse(parts[2] == "", 0, map[parts[2]])
        return(tens * 10 + ones)
      }
      map[f]
    })
    
    numeric_floors <- numeric_floors[!is.na(numeric_floors)]
    
    if(length(numeric_floors) != 1) return(NA_real_)
    
    numeric_floors
  })
}


# Convert "有"/"無" to TRUE/FALSE
char_to_logical <- function(x) if_else(x == "有", TRUE, FALSE)

# Apply transformations
housing <- housing %>%
  mutate(
    management = char_to_logical(management),
    elevator = char_to_logical(elevator),
    totalPriceL = log(totalPrice),
    balconyAreaL = log(balconyArea),
    buildingAreaL = log(buildingArea),
    parkingAreaL = log(parkingArea),
    parkingPriceL = log(parkingPrice),
    sqmPriceL = log(sqmPrice),
    totalFloors = chinese_to_int(totalFloors),
    apartmentFloor = chinese_to_int(apartmentFloor)
  )


# Save processed data
write_csv(housing, "data/PROCESSED_new_taipei_housing_transactions_2018_2023.csv")