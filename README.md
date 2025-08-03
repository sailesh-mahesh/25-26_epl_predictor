Here’s your `README.md` file with all emojis removed:

```markdown
# Premier League Table Predictor (2025/26)

This project is a simple, data-driven model to predict the final standings of the **2025/26 Premier League** season. It was built using Python and machine learning techniques, combining historical football data with manual transfer window ratings. It’s my first proper project, so the focus is on building something that sligns with my interests

---

## What It Does

- **Predicts final EPL standings** based on stats from past seasons
- Uses **Random Forest Regression** to estimate points, goals scored, and goals conceded
- Lets you rate each team’s **summer transfer window** (+5 to -5) to factor in new signings and departures

---

## Features

- **Historical Match Data**: Premier League + Championship from the last 5 seasons
- **Calculated Team Stats**:
  - Points, Wins, Draws, Losses
  - Goals Scored, Goals Conceded, Goal Difference
  - Team Form (last 10 matches of previous season)
  - xG and xGA (mock data for now)
  - Long-term PL averages for context
- **Transfer Impact Input**: You get to manually rate each team’s transfer window
- **Promoted/Relegated Flags**: Adjusts for newly promoted or relegated clubs

---

## How It Works

1. **Preprocessing** – Cleans and combines all the raw CSV files
2. **Feature Engineering** – Builds performance stats, flags, and xG features
3. **Model Training** – Trains a Random Forest Regressor and asks for your transfer ratings
4. **Prediction** – Outputs a full Premier League table with predicted results

---

## Project Structure

```

premier-league-predictor/
├── data/
│   ├── premier\_league/       # Raw EPL match CSVs (E0\_*.csv)
│   ├── championship/         # Raw Championship match CSVs (E1\_*.csv)
│   ├── combined\_data.csv     # After preprocessing
│   └── final\_features\_complete.csv  # After feature engineering
└── code/
├── 01\_data\_preprocessing.py
├── 02\_feature\_engineering.py
└── 03\_model\_training.py

````

---

## Setup

1. **Clone the repo**:
```bash
git clone <your-repo-url>
cd premier-league-predictor
````

2. **(Optional) Create a virtual environment**:

```bash
python3 -m venv epl_pred_venv
source epl_pred_venv/bin/activate  # Windows: .\epl_pred_venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install pandas scikit-learn numpy
```

4. **Download match data**:

* Visit [football-data.co.uk](https://football-data.co.uk/englandm.php)
* Download EPL (`E0`) and Championship (`E1`) CSVs for seasons 2020/21 to 2024/25
* Save them in:

  * `data/premier_league/`
  * `data/championship/`

5. **(Optional) Replace mock xG data**:

* Real xG/xGA data can be scraped from [Understat](https://understat.com/) or found on Kaggle
* Replace the mock values inside `code/02_feature_engineering.py`

---

## Running the Model

1. Run data preprocessing:

```bash
python code/01_data_preprocessing.py
```

2. Run feature engineering:

```bash
python code/02_feature_engineering.py
```

3. Train model and make predictions:

```bash
python code/03_model_training.py
```

You'll be prompted to enter transfer ratings for each team (e.g., +5 = excellent window, -5 = terrible).

---

## Dependencies

* Python 3.x
* pandas
* scikit-learn
* numpy

---

## Notes

This is a learning project, so there’s plenty of room for improvement. Things like more advanced models, better xG data, squad value integration, and automated transfer evaluation could all be added down the line.

---

## Feedback

Feel free to fork, use, or suggest improvements. Any feedback is welcome!


