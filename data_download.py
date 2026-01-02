import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import main as mt

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Убираем мультииндекс если он есть
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Проверяем что данные загружены
    if len(data) == 0:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")
    
    # Сохраняем с индексом (Date) - каждый тикер в свой файл
    csv_filename = f'tickers/{ticker}.csv'
    data.to_csv(csv_filename, index=True, encoding='utf-8')
    
    print(f"✓ DataFrame успешно записан в {csv_filename}")
    print(f"  Колонки: {data.columns.tolist()}")
    print(f"  Период: {data.index.min()} - {data.index.max()}")
    print(f"  Строк: {len(data)}")
    print(f"{'='*60}\n")
