import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from data_feed import load_data
from back_test import run_portfolio_backtest
import os

# Кризисные периоды для тестирования
crisis_times_dates = [
    ("1999-12-01", "2025-12-01")  # standard full period
#    ("2000-01-01", "2002-12-01"),  # доткомы
#    ("2007-01-01", "2009-12-01"),  # финансовый кризис
#    ("2020-01-01", "2020-12-01"),  # COVID-19
#    ("2022-01-01", "2022-12-01")   # 2022 медвежий рынок
]

# Базовые параметры
portfolio_tickers = ["SPY", "QQQ", "GLD", "USO"]
start_date = "1999-01-01"
end_date = "2025-01-01"
goal_volatility = 0.10  # целевая волатильность портфеля
max_asset_risk = 0.033   # максимальный риск на один актив
numeraire = 'usd'  # 'usd' или 'gold' - валюта отображения графиков
commission = 0.002  # 0.2% комиссия за маркет ордеры (вход и выход)


# Конфигурация для moving averages и ATR
ws = 20   # window for short SMA
wl = 200  # window for long SMA
watr = 20 # window for ATR
slope_lookback = 5  # период для проверки роста SMA (1 = сравнение с предыдущим днем, 5 = с 5 дней назад)
different_modes = 1 # количество разных режимов для тестирования
# Запуск базового бэктеста
if __name__ == "__main__":
    # Портфельная торговля (единственный режим)
     # тестируем разные значения slope_lookback
    print("\n" + "="*80)
    print("ПОРТФЕЛЬНАЯ ТОРГОВЛЯ (RISK PARITY / VOLATILITY TARGETING)")
    print("="*80 + "\n")
    
    crisis_results = []
    for i in range(different_modes):
        print(f"\n--- Портфельный тест {i+1} на кризисных периодах ---")
        for (crisis_start, crisis_end) in crisis_times_dates:
            result = run_portfolio_backtest(
                tickers=portfolio_tickers,
                start_date=crisis_start,
                end_date=crisis_end,
                ws=ws,
                wl=wl,
                watr=watr,
                capital=100000,  # разный стартовый капитал для различения графиков
                slope_lookback=slope_lookback,
                allocation='risk_parity',  # RISK PARITY вместо равного распределения
                volatility_target=goal_volatility,  # целевая волатильность портфеля
                volatility_lookback=252,  # год для расчета волатильности
                show_individual_plots=False,
                show_portfolio_plot=True,
                warm_up_days=400,
                normalize_by=numeraire,  # отображение в золоте или USD
                commission=commission  # комиссия за маркет ордеры
            )
        crisis_results.append(result)
    print("="*80 + "\n")

#merge all results from backtests into one .txt file
#they all have the save name "metrics_{counter}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
#merge ALL .txt files in directory into one file "all_crisis_results.txt"

    output_file = "all_crisis_results.txt"

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir("."):
            if filename.endswith(".txt") and filename != output_file:
                try:
                    with open(filename, "r", encoding="utf-8") as infile:
                        outfile.write(f"\n--- Содержимое файла: {filename} ---\n")
                        outfile.write(infile.read())
                except Exception as e:
                    print(f"Ошибка при чтении {filename}: {e}")
