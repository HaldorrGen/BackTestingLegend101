import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def load_data(ticker, start_date, end_date, ws, wl, watr, warm_up_days=0):
    """
    Загружает данные из CSV и применяет срез по периоду
    
    A) Источник данных: читаем полный CSV
    B) Срез по периоду: фильтруем по start_date и end_date
    
    Args:
        warm_up_days: количество торговых дней для прогрева индикаторов перед start_date
    """
    # A) ИСТОЧНИК ДАННЫХ - загружаем полный датасет из CSV
    csv_filename = f'tickers/{ticker}.csv'
    
    try:
        data = pd.read_csv(csv_filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Файл {csv_filename} не найден. "
            f"Запустите data_download.py для скачивания данных {ticker}"
        )
    
    # Парсим дату и устанавливаем как индекс
    # Проверяем разные варианты названий колонки с датой
    date_column = None
    for col in ['Date', 'date', 'DATE', 'Unnamed: 0']:
        if col in data.columns:
            date_column = col
            break
    
    if date_column:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column)
    elif not isinstance(data.index, pd.DatetimeIndex):
        # Если индекс уже есть, но не datetime, пытаемся его преобразовать
        try:
            data.index = pd.to_datetime(data.index)
        except:
            pass
    
    # Убедимся, что индекс - это DatetimeIndex и отсортирован
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Не удалось преобразовать индекс в DatetimeIndex")
    
    # Сортируем по времени (если вдруг не отсортировано)
    data = data.sort_index()
    
    # Ensure we have a proper DataFrame with single-level columns
    # не ебу почему, но иногда yfinance возвращает мультииндекс
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Проверяем наличие необходимых колонок
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")
    
    # B) СРЕЗ ПО ПЕРИОДУ - фильтруем данные по датам с учетом warm-up
    actual_start_date = None
    if start_date is not None:
        actual_start_date = pd.to_datetime(start_date)
        
        # Если нужен warm-up период, берем данные раньше
        if warm_up_days > 0:
            # Находим индекс start_date в данных
            data_before_start = data[data.index < actual_start_date]
            if len(data_before_start) >= warm_up_days:
                # Берем warm_up_days торговых дней назад
                warm_up_start = data_before_start.index[-warm_up_days]
                data = data[data.index >= warm_up_start]
            else:
                # Если недостаточно данных для warm-up, берем все что есть
                pass
        else:
            data = data[data.index >= actual_start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data[data.index <= end_date]
    
    if len(data) == 0:
        raise ValueError(f"После фильтрации по периоду {start_date} - {end_date} данных не осталось")
    
    # ---- СКОЛЬЗЯЩИЕ СРЕДНИЕ ----
    data["SMA_short"] = data["Close"].rolling(window=ws).mean()
    data["SMA_long"]  = data["Close"].rolling(window=wl).mean()

    # ---- ATR ----
    high_low   = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close  = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(window=watr).mean()
    
    # Добавляем метку где начинается реальный период (после warm-up)
    if warm_up_days > 0 and actual_start_date is not None:
        data["is_analysis_period"] = data.index >= actual_start_date
    else:
        data["is_analysis_period"] = True

    return data 