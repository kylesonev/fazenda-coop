# little-farm

This is my first data science project. There’s still a lot to learn, but I’m proud of the progress so far.

As part of the dataset, I used real data from a farm located in my hometown — **Iguape/SP, Brazil**.  
This farm produces:

- Milk
- Cheese
- Dulce de leche
- Pork
- Eggs
- Vegetables

The goal is to analyze and forecast the monthly revenue of these products, with a focus on **milk**, the farm’s main product, using time series models such as **Prophet** and **Auto ARIMA**.

---

## Objective

Forecast **monthly milk revenue** using historical data and **exogenous variables**, such as sales of eggs, pork, cheese, etc.

---

## Project Structure

```
data/         # Raw CSV data
notebooks/    # Exploratory analysis and modeling in Jupyter
src/          # Python scripts (preprocessing, modeling, metrics)
requirements.txt  # Project dependencies
```

---

## 🚀 How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/kylesonev/fazenda-coop.git
cd fazenda-coop
```

2. **Create and activate a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the notebooks** in the `notebooks/` folder to explore the data and results.

---

## ✨ Models Used

- **Prophet** (Meta/Facebook)
- **Auto ARIMA** (pmdarima)
- Evaluation metrics: **RMSE**, **MAE**, **MAPE**

---

## 🛠️ Work in Progress

- [ ] Refactor notebooks into reusable scripts (`src/`)
- [ ] Add automated tests
- [ ] Improve interactive visualizations
